'''
Based on:
- https://github.com/Project-MONAI/tutorials/blob/main/3d_classification/densenet_training_array.ipynb
- https://github.com/spinalcordtoolbox/disc-labeling-hourglass/blob/main/src/dlh/train/main.py 
'''

import logging
import os
import sys
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset, ArrayDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)

from ply.utils.utils import fetch_array_from_config_classifier, tuple_type

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 database format.')
    parser.add_argument('--config', required=True, help='Config JSON file where every label used for TRAINING, VALIDATION and TESTING has its path specified ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--path-out', required=True, help='Path to output directory. Example: ~/data/dataset-nnunet (Required)')
    parser.add_argument('--path-out', required=True, help='Path to output directory. Example: ~/data/dataset-nnunet (Required)')
    parser.add_argument('--use-lock-fov', action='store_true', help='Use random locked fov. See fov for window size (default=False)')
    parser.add_argument('--fov', type=tuple_type, default=(150,150), help='Fov size if --use-lock-fov is True (default=(150,150))')
    parser.add_argument('--dim', type=str, default='3D', choices=['2D', '3D'], help='Input imensions. Default=3D. Choices=["2D", "3D"]')
    parser.add_argument('--weight-folder', type=str, default=os.path.abspath('src/ply/weights/classifiers'),
                        help='Folder where the classifiers weights will be stored and loaded. Will be created if does not exist. (default="src/ply/weights/classifiers")')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Load variables
    weight_folder = args.weight_folder

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

    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Create weights folder to store training weights
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)
    
    # Load config data
    # Read json file and create a dictionary
    with open(args.config_data, "r") as file:
        config_data = json.load(file)
    
    # Load images for training and validation
    print('loading images...')
    imgs_train, masks_train, discs_labels_train, subjects_train, res_train, _ = fetch_array_from_config_classifier(config_data=config_data,
                                                                                                                   fov=args.fov if args.use_lock_fov else None,
                                                                                                                   dim=args.dim,
                                                                                                                   split='TRAINING')
    
    imgs_val, masks_val, discs_labels_val, subjects_val, res_val, _ = fetch_array_from_config_classifier(config_data=config_data,
                                                                                                         dim=args.dim,
                                                                                                         split='VALIDATION')

    # Define transforms
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])

    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

    # Define nifti dataset, data loader
    check_ds = ImageDataset(image_files=images, labels=labels, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=3, num_workers=2, pin_memory=pin_memory)

    im, label = monai.utils.misc.first(check_loader)
    print(type(im), im.shape, label, label.shape)

    # create a training data loader
    train_ds = ImageDataset(image_files=images[:10], labels=labels[:10], transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=pin_memory)

    # create a validation data loader
    val_ds = ImageDataset(image_files=images[-10:], labels=labels[-10:], transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=pin_memory)


if __name__=='__main__':
    main()