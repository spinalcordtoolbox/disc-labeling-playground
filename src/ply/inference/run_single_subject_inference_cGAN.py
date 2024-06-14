"""
This Python script applies a T2-weighted (T2w) contrast transformation to the input image and then saves the output image using the flag --path-out.

Author: Nathan Molinier
"""

import os
import sys
import shutil
import numpy as np
import argparse
import random
import json
from tqdm import tqdm

import torch
import torch.optim as optim

import monai
from monai.inferers import sliding_window_inference
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
    CenterScaleCropd
)

from ply.utils.load_image import fetch_and_preproc_image_cGAN_NoSeg
from ply.utils.image import Image, zeros_like
from ply.utils.utils import tmp_create


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run cGAN inference on a single subject')
    parser.add_argument('--path-in', type=str, required=True, help='Path to the input image (Required)')
    parser.add_argument('--path-out', type=str, default='', help='Output path after inference: (Default= --path-in folder)')
    parser.add_argument('--weight-path', type=str, required=True, help='Path to the network weights. (Required)')
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
    path_in = os.path.abspath(args.path_in)
    path_out = path_in.replace('.nii.gz','')+'_desc-crop_fakeT2w.nii.gz' if not args.path_out else args.path_out
    weight_path = os.path.abspath(args.weight_path)

    # Check if weight path exists
    if not os.path.exists(weight_path):
        raise ValueError(f'Weights path {weight_path} does not exist')
    
    # Create temp directory for preprocessing
    tmpdir = tmp_create(basename='cGAN-Preproc')
    
    # Load images for inference
    print('-'*40)
    print('Loading image with preprocessing')
    print('-'*40)
    img_list = fetch_and_preproc_image_cGAN_NoSeg(path_in=path_in, tmpdir=tmpdir)

    # Define test transforms
    crop_size = tuple(map(int, args.weight_path.split('cropRSP_')[-1].split('_')[0].split('-'))) # RSP
    pixdim=tuple(map(float, args.weight_path.split('pixdimRSP_')[-1].split('_')[0].split('-')))
    if 'scaleCrop_' in  args.weight_path:
        scale_crop=tuple(map(float, args.weight_path.split('scaleCrop_')[-1].split('_')[0].split('-')))
    else:
        scale_crop = (1,1,1)
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="LIA"), # RSP --> LIA
            Spacingd(
                keys=["image"],
                pixdim=pixdim,
                mode=2, # spline interpolation
            ),
            CenterScaleCropd(keys=["image"], roi_scale=scale_crop,),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=crop_size,),
            LabelToContourd(keys=["image"], kernel_type='Laplace'),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ]
    )

    inv_transforms = Compose([
                EnsureTyped(keys=["pred"]),
                Invertd(keys=["pred"], transform=test_transforms, 
                        orig_keys=["image"], 
                        nearest_interp=False, to_tensor=True),
        ])

    # Define test dataset
    test_ds = CacheDataset(
                            data=img_list,
                            transform=test_transforms,
                            cache_rate=0.25,
                            num_workers=4,
                            )

    # Define test DataLoader
    data_loader = DataLoader(
                            test_ds, 
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
    data_iterator = tqdm(data_loader, desc="Run inference", dynamic_ncols=True)
    for step, batch in enumerate(data_iterator):
        # Load input
        x = batch["image"].to(device)

        # Use sliding_window_inference from MONAI to deal with bigger images
        y_fake = generator(x)
        # y_fake = sliding_window_inference(x,
        #                                   crop_size, 
        #                                   sw_batch_size=3, 
        #                                   predictor=generator, 
        #                                   overlap=0.1, 
        #                                   progress=False)

        # Transform output to its original shape/resolution
        batch["pred"] = y_fake.data.cpu()

        batch = [inv_transforms(i) for i in decollate_batch(batch)][0]
        out_fake = batch["pred"].numpy()[0]

        # Load input path and fetch path information
        input_img = Image(path_in)
        original_orientation = input_img.orientation
        input_crop_img = Image(img_list[step]['image'])

        # TODO: Reshape image to its original shape

        # Create output folder if does not exists
        out_folder = os.path.dirname(path_out)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # Save output
        out_img = zeros_like(input_crop_img)
        out_img.data = out_fake
        out_img.change_orientation(original_orientation)
        out_img.save(path_out)

    # Remove tempdir
    print('Removing temp directory...')
    shutil.rmtree(tmpdir)

    print('-'*40)
    print(f'Inference done: {path_out} was created')
    print('-'*40)

if __name__=='__main__':
    main()