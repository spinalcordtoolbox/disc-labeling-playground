import logging
import os
import sys
import shutil
import numpy as np
import pandas as pd
import argparse
import random
import json
import wandb
import copy
from tqdm import tqdm
import subprocess
import nibabel as nib

import torch
import torch.optim as optim

import monai
from monai.data import DataLoader, CacheDataset, Dataset, decollate_batch
from monai.networks.nets import UNet, AttentionUnet
from monai.transforms import (
    LoadImaged,
    Orientationd,
    EnsureChannelFirstd,
    Spacingd,
    Compose,
    NormalizeIntensityd,
    ResizeWithPadOrCropd,
    LabelToContourd,
    Invertd,
    EnsureTyped,
    FromMetaTensord
)

from ply.data_management.utils import fetch_subject_and_session
from ply.utils.load_image import fetch_and_preproc_image_cGAN
from ply.utils.image import Image, zeros_like
from ply.utils.utils import tmp_create


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run cGAN inference on a single subject')
    parser.add_argument('--path-in', type=str, required=True, help='Path to the input image (Required)')
    parser.add_argument('--path-seg', type=str, required=True, help='Path to the input spinal cord segmentation (Required)')
    parser.add_argument('--ofolder', type=str, required=True, help='Output folder (Required)')
    parser.add_argument('--weight-path', type=str, required=True, help='Path to the network weights. (Required')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Use cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Set seed
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed) 
    
    # Load variables
    path_in = os.path.abspath(args.path_in)
    path_seg = os.path.abspath(args.path_seg)
    out_folder = os.path.abspath(args.ofolder)
    weight_path = os.path.abspath(args.weight_path)

    # Check if out_folder path exists
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Check if weight path exists
    if not os.path.exists(weight_path):
        raise ValueError(f'Weights path {weight_path} does not exist')
    
    # Create temp directory for preprocessing
    tmpdir = tmp_create(basename='cGAN-Preproc')
    
    # Load images for inference
    print('-'*40)
    print('Loading image with preprocessing')
    print('-'*40)
    img_list = fetch_and_preproc_image_cGAN(path_in=path_in, path_seg=path_seg, tmpdir=tmpdir)

    # Define test transforms
    crop_size = tuple(map(int, args.weight_path.split('cropRSP_')[-1].split('_')[0].split('-'))) # RSP
    pixdim=tuple(map(float, args.weight_path.split('pixdimRSP_')[-1].split('_')[0].split('-')))
    interpolation=2 # spline
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="LIA"), # RSP- --> LIA+
            Spacingd(
                keys=["image"],
                pixdim=pixdim,
                mode=(interpolation),
            ),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=crop_size,),
            LabelToContourd(keys=["image"], kernel_type='Laplace'),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ]
    )

    split_test_transforms1 = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="LIA"), # RSP- --> LIA+
            Spacingd(
                keys=["image"],
                pixdim=pixdim,
                mode=("bilinear"),
            )
        ]
    )

    split_test_transforms2 = Compose(
        [
            LabelToContourd(keys=["image"], kernel_type='Laplace'),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ]
    )

    def inv_transform(orig_size):
        inv_transforms = Compose([
                    EnsureTyped(keys=["pred"]),
                    Invertd(keys=["pred"], transform=split_test_transforms2, 
                            orig_keys=["image"], 
                            nearest_interp=False, to_tensor=True),
                    ResizeWithPadOrCropd(keys=["pred"], spatial_size=orig_size,),
            ])
        return inv_transforms

    # Define test dataset
    test_ds = CacheDataset(
                            data=img_list,
                            transform=test_transforms,
                            cache_rate=0.25,
                            num_workers=4,
                            )

    trans1_ds = CacheDataset(
                            data=img_list,
                            transform=split_test_transforms1,
                            cache_rate=0.25,
                            num_workers=4,
                            )

    # Define test DataLoader
    test_loader = DataLoader(
                            test_ds, 
                            batch_size=1,
                            shuffle=False, 
                            num_workers=4, 
                            pin_memory=True, 
                            persistent_workers=True
                            )

    trans1_loader = DataLoader(
                            trans1_ds, 
                            batch_size=1,
                            shuffle=False, 
                            num_workers=4, 
                            pin_memory=True, 
                            persistent_workers=True
                            )

    # Create generator model
    generator = AttentionUnet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                kernel_size=3).to(device)
    
    # Load network weights
    generator.load_state_dict(torch.load(weight_path, map_location=torch.device(device))["generator_weights"])
    generator.eval()

    # Start inference
    print('-'*40)
    print('Starting inference')
    print('-'*40)
    data_iterator = tqdm(test_loader, desc="Run inference", dynamic_ncols=True)
    for step, (batch, batch1) in enumerate(zip(data_iterator, trans1_loader)):
        # Load input
        x = batch["image"].to(device)

        # Use sliding_window_inference from MONAI to deal with bigger images
        y_fake = generator(x)
        batch["pred"] = y_fake.data.cpu()

        batch1 = [inv_transform(batch1["image"].shape[2:])(i) for i in decollate_batch(batch)][0]

        affine = batch1['pred'].meta['affine'] # Affine of the upsampled image

        # Save prediction in its upsampled format
        #path_cGAN_pred = img_list[step]['image'].replace('.nii.gz', '_fakeT2w.nii.gz')
        path_cGAN_pred = os.path.join(out_folder, os.path.basename(img_list[step]['image']).replace('.nii.gz', '_fakeT2w.nii.gz'))
        nib.save(nib.Nifti1Image(batch1["pred"].numpy()[0], affine), path_cGAN_pred)

        # Run SPINEPS prediction
        subprocess.check_call([
            '/home/GRAMES.POLYMTL.CA/p118739/.conda/envs/spineps_env/bin/python',
            '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/code/spineps/simple_run.py',
            '--path-in', path_cGAN_pred,
            '--ofolder', out_folder
        ])

    # Remove tempdir
    print('Removing temp directory...')
    shutil.rmtree(tmpdir)

    print('-'*40)
    print(f'Inference done: {out_folder} was created')
    print('-'*40)

if __name__=='__main__':
    main()