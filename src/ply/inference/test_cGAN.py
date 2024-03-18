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

import torch
import torch.optim as optim

import monai
from monai.data import DataLoader, CacheDataset
from monai.networks.nets import UNet
from monai.transforms import (
    LoadImaged,
    Orientationd,
    EnsureChannelFirstd,
    Spacingd,
    Compose,
    ResizeWithPadOrCropd,
    RandRotate90d,
    RandFlipd,
    NormalizeIntensityd
)

from ply.data_management.utils import fetch_subject_and_session
from ply.utils.load_config import fetch_and_preproc_config_cGAN
from ply.utils.image import Image, zeros_like


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 database format.')
    parser.add_argument('--config', required=True, help='Config JSON file where every label used for TRAINING, VALIDATION and TESTING has its path specified ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--contrast', type=str, default='T1w', help='Input contrast that was used for training (default="T1w").')
    parser.add_argument('--weight-path', required=True, type=str, help='Path to the network weights. (default="src/ply/weights/3DGAN")')
    parser.add_argument('--out-derivative', default='derivatives/fakeT2w', type=str, help='Derivative folder where the output data will be stored. (default="derivatives/fakeT2w")')
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
    
    # Load config data
    # Read json file and create a dictionary
    with open(args.config, "r") as file:
        config_data = json.load(file)
    
    # Load variables
    weight_path = args.weight_path
    out_derivative = args.out_derivative

    # Check if weight path exists
    if not os.path.exists(weight_path):
        raise ValueError(f'Weights path {weight_path} does not exist')
    
    # Load images for testing
    print('loading images...')
    test_list, err_test = fetch_and_preproc_config_cGAN(
                                            config_data=config_data,
                                            cont=args.contrast,
                                            split='TESTING'
                                            )
    
    # Define test transforms
    crop_size = (192, 256, 192) # RSP or LIA
    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="LIA"), # RSP --> LIA
            Spacingd(
                keys=["image", "label"],
                pixdim=(1., 1., 1.),
                mode=("bilinear", "nearest"),
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=crop_size,),
        ]
    )

    # Define test dataset
    test_ds = CacheDataset(
                            data=test_list,
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
    generator = UNet(
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
    data_iterator = tqdm(data_loader, desc="Run inference", dynamic_ncols=True)
    for step, batch in enumerate(data_iterator):
        # Load input
        x = batch["image"].to(device)

        # Get output from generator
        y_fake = generator(x)

        # Load input path and fetch path information
        input_path = test_list[step]["image"]
        input_img = Image(input_path)
        original_orientation = input_img.orientation
        input_img.change_orientation('RSP')
        nx, ny, nz, nt, px, py, pz, pt = input_img.dim
        subjectID, sessionID, filename, contrast, echoID, acq = fetch_subject_and_session(input_path)

        # Create output folder if does not exists
        out_folder = os.path.join(input_path.split(subjectID)[0].split('derivatives')[0], out_derivative, subjectID, sessionID, contrast)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # Transform output to its original shape/resolution
        y_fake = y_fake.data.cpu()[0]
        transform_dict = {"image":y_fake}
        out_transforms = Compose(
        [
            Spacingd(
                keys=["image"],
                pixdim=(px, py, pz),
                #mode=("bilinear", "nearest"),
            ),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=(nx, ny, nz),),
        ]
        )
        transform_out = out_transforms(transform_dict)
        out_fake = transform_out["image"].numpy()

        # Save output
        out_fname = os.path.basename(test_list[step]["label"]).split('.nii.gz')[0] + '_desc-fake.nii.gz' # Create filename
        out_path = os.path.join(out_folder, out_fname)
        out_img = zeros_like(input_img)
        out_img.data = out_fake
        out_img.change_orientation(original_orientation)
        out_img.save(out_path)
    

if __name__=='__main__':
    main()