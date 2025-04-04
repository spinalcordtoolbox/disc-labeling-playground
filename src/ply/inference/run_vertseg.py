"""
This Python script applies the model VertSeg to a given input image then saves the output segmentation using the flag --path-out.

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
from monai.transforms import (
    LoadImaged,
    Orientationd,
    EnsureChannelFirstd,
    Spacingd,
    Compose,
    NormalizeIntensityd,
    ResizeWithPadOrCropd,
    Invertd,
    EnsureTyped,
)

from ply.utils.load_config import fetch_data_config
from ply.utils.image import Image, zeros_like
from ply.utils.utils import tmp_create
from ply.models.transform import RandLabelToContourd
from ply.models.segmentation.attunet import AttentionUnet


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run cGAN inference on a single subject')
    parser.add_argument('--path-in', type=str, required=True, help='Path to the input image or config.json files with image paths in TESTING field (Required)')
    parser.add_argument('--path-out', type=str, default='', help='Output path after inference: (Default= --path-in folder)')
    parser.add_argument('--weight-path', type=str, required=True, help='Path to the network weights. (Required)')
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
    path_out = path_in.replace('.nii.gz','_seg.nii.gz') if not args.path_out else args.path_out
    weight_path = os.path.abspath(args.weight_path)

    # Check if weight path exists
    if not os.path.exists(weight_path):
        raise ValueError(f'Weights path {weight_path} does not exist')
    
    # Load images for inference
    print('-'*40)
    print('Loading image with preprocessing')
    print('-'*40)
    if path_in.endswith(".json"):
        # Load JSON
        with open(args.config, "r") as file:
            config_data = json.load(file)
        
        test_list, err_train = fetch_data_config(
                config_data=config_data,
                split='TESTING'
            )
    else:
        test_list = [{'image':path_in}]

    # Define test transforms
    crop_size = tuple(map(int, args.weight_path.split('cropRSP_')[-1].split('_')[0].split('-'))) # RSP
    pixdim=tuple(map(float, args.weight_path.split('pixdimRSP_')[-1].split('_')[0].split('-')))

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
            RandLabelToContourd(keys=["image"], kernel_type="Scharr", prob=1),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ]
    )

    inv_transforms = Compose(
        [
            EnsureTyped(keys=["pred"]),
            Invertd(keys=["pred"], transform=test_transforms, 
                    orig_keys=["image"], 
                    nearest_interp=False, to_tensor=True),
        ])

    # Define test dataset
    test_ds = CacheDataset(
                            data=test_list,
                            transform=test_transforms,
                            cache_rate=0,
                            num_workers=4,
                            )

    # Define test DataLoader
    data_loader = DataLoader(
                            test_ds, 
                            batch_size=1,
                            shuffle=False, 
                            num_workers=4, 
                            pin_memory=False, 
                            persistent_workers=False
                            )

    # Create generator model
    model = AttentionUnet(
                spatial_dims=3,
                in_shape=crop_size,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2, 2),
                kernel_size=3).to(device)
    
    # Load network weights
    model.load_state_dict(torch.load(weight_path, map_location=torch.device(device))["weights"])
    model.eval()

    # Start inference
    print('-'*40)
    print('Starting inference')
    print('-'*40)
    data_iterator = tqdm(data_loader, desc="Run inference", dynamic_ncols=True)
    for step, batch in enumerate(data_iterator):
        # Load input
        x = batch["image"].to(device)

        # Use sliding_window_inference from MONAI to deal with bigger images
        y_pred = inference(x, model, crop_size)

        # Transform output to its original shape/resolution
        batch["pred"] = y_pred.data.cpu()

        batch = [inv_transforms(i) for i in decollate_batch(batch)][0]
        pred = batch["pred"].numpy()[0]

        # Load input path and fetch path information
        input_img = Image(path_in)
        original_orientation = input_img.orientation

        # TODO: Reshape image to its original shape

        # Create output folder if does not exists
        out_folder = os.path.dirname(path_out)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # Save output
        out_seg = zeros_like(input_img).change_orientation('RSP')
        out_seg.data = pred
        out_seg.change_orientation(original_orientation)
        out_seg.save(path_out)

        print('-'*40)
        print(f'Inference done: {path_out} was created')
        print('-'*40)


def inference(input, model, crop_size):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=crop_size,
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
        )

    # Run model
    with torch.autocast("cuda"):
        return _compute(input)
    

if __name__=='__main__':
    main()