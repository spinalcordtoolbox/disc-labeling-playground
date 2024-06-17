"""
This Python script operates based on a JSON configuration file to perform the following tasks:

1. T2w Contrast Translation: Applies a T2-weighted (T2w) contrast transformation to each input image.
2. SPINEPS Inference: Executes SPINEPS inference on the generated synthetic contrast.
3. Image Saving: Stores all processed images within a designated derivatives folder.

Author: Nathan Molinier
"""

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
    FromMetaTensord,
    CenterScaleCropd
)

from ply.data_management.utils import fetch_subject_and_session
from ply.utils.load_config import fetch_image_config_cGAN
from ply.utils.image import Image, zeros_like
from ply.utils.utils import tmp_create


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run cGAN and SPINEPS inference on a JSON config file')
    parser.add_argument('--config', required=True, help='Config JSON file where every image path used for inference must appear in the field TESTING ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--out-derivative', default='derivatives/label-SPINEPSwithcGAN', type=str, help='Derivative folder where the output data will be stored. (default="derivatives/label-SPINEPSwithcGAN")')
    parser.add_argument('--weight-path', type=str, required=True, help='Path to the network weights. (Required')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Use cuda
    device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")

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
    config_path = os.path.abspath(args.config)
    derivatives_folder = args.out_derivative
    weight_path = os.path.abspath(args.weight_path)
    qc = True

    # Load config data
    # Read json file and create a dictionary
    with open(config_path, "r") as file:
        config_data = json.load(file)

    # Check if weight path exists
    if not os.path.exists(weight_path):
        raise ValueError(f'Weights path {weight_path} does not exist')
    
    # Load images for inference
    print('-'*40)
    print('Loading image with preprocessing')
    print('-'*40)
    img_list, err = fetch_image_config_cGAN(config_data=config_data,
                                       split='TESTING')

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

        # Create output folder
        path_image = img_list[step]['image']
        bids_path = path_image.split('sub-')[0]
        derivatives_path = os.path.join(bids_path, derivatives_folder)
        out_folder = os.path.join(derivatives_path, os.path.dirname(path_image.replace(bids_path,'')))
        if not os.path.exists(out_folder):
            print(f'{out_folder} was created')
            os.makedirs(out_folder)

        # Save prediction in its upsampled format
        path_cGAN_pred = os.path.join(out_folder, os.path.basename(path_image).replace('.nii.gz', '_fakeT2w.nii.gz'))
        nib.save(nib.Nifti1Image(batch1["pred"].numpy()[0], affine), path_cGAN_pred)

        # Run SPINEPS prediction
        subprocess.check_call([
            '/home/GRAMES.POLYMTL.CA/p118739/.conda/envs/spineps_env/bin/python',
            '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/code/spineps/simple_run.py',
            '--path-in', path_cGAN_pred,
            '--ofolder', out_folder
        ])

        # Register the vertebrae segmentation to the original image using only q-form and s-form
        spine_path = os.path.join(out_folder, os.path.basename(path_image).replace('.nii.gz', '_fakeT2w_label-vert_dseg.nii.gz'))
        subprocess.run(['sct_register_multimodal',
                                '-i', spine_path,
                                '-d', path_image,
                                '-identity', '1', # Linear registration based on q-form and s-form
                                '-x', 'nn',
                                '-o', spine_path])

        # Generate QC
        if qc:
            qc_path = os.path.join(derivatives_path, 'qc')
            subprocess.check_call([
                                "sct_qc", 
                                "-i", path_image,
                                "-s", spine_path,
                                "-d", spine_path,
                                "-p", "sct_deepseg_lesion",
                                "-plane", "sagittal",
                                "-qc", qc_path
                            ])

    print('-'*40)
    print(f'Inference done: {out_folder} was created')
    print('-'*40)

if __name__=='__main__':
    main()