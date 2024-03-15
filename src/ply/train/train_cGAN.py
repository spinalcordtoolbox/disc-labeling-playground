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

from ply.utils.utils import tuple_type
from ply.models.discriminator import Discriminator
from ply.utils.load_config import fetch_and_preproc_config_cGAN


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 database format.')
    parser.add_argument('--config', required=True, help='Config JSON file where every label used for TRAINING, VALIDATION and TESTING has its path specified ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--contrast', type=str, default='T1w', help='Input contrast that will be used for training (default="T1w").')
    parser.add_argument('--batch-size', type=int, default=3, help='Training batch size (default=3).')
    parser.add_argument('--nb-epochs', type=int, default=500, help='Number of training epochs (default=500).')
    parser.add_argument('--start-epoch', type=int, default=0, help='Starting epoch (default=0).')
    parser.add_argument('--alpha', type=int, default=100, help='L1 loss multiplier (default=100).')
    parser.add_argument('--weight-folder', type=str, default=os.path.abspath('src/ply/weights/3D-CGAN'),
                        help='Folder where the cGAN weights will be stored and loaded. Will be created if does not exist. (default="src/ply/weights/3DGAN")')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Use cuda
    pin_memory = torch.cuda.is_available()
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
    weight_folder = args.weight_folder
    in_contrast = args.contrast
    out_contrast = config_data['CONTRASTS']

    # Create weights folder to store training weights
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)

    if len(out_contrast.split('_'))>1:
        raise ValueError(f'Multiple output contrast detected, check data config["CONTRAST"]={out_contrast}')
    
    # Load images for training and validation
    print('loading images...')
    train_list, err_train = fetch_and_preproc_config_cGAN(
                                            config_data=config_data,
                                            cont=args.contrast,
                                            split='TRAINING'
                                            )
    
    val_list, err_val = fetch_and_preproc_config_cGAN(
                                            config_data=config_data,
                                            cont=args.contrast,
                                            split='VALIDATION'
                                            )
    
    # Define transforms
    crop_size = (192, 256, 192) # RSP
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RSP"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1., 1., 1.),
                mode=("bilinear", "nearest"),
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=crop_size,),
            #LabelToContour(kernel_type='Laplace'), # TODO
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            NormalizeIntensityd(
                keys=["image", "label"], 
                nonzero=False, 
                channel_wise=False),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RSP"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1., 1., 1.),
                mode=("bilinear", "nearest"),
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=crop_size,),
            NormalizeIntensityd(
                keys=["image", "label"], 
                nonzero=False, 
                channel_wise=False)
        ]
    )

    # Define train and val dataset
    train_ds = CacheDataset(
                            data=train_list,
                            transform=train_transforms,
                            cache_rate=0.25,
                            num_workers=4,
                            )
    val_ds = CacheDataset(
                        data=val_list,
                        transform=val_transforms,
                        cache_rate=0.25,
                        num_workers=4,
                        )

    # Define train and val DataLoader
    train_loader = DataLoader(
                            train_ds, 
                            batch_size=args.batch_size,
                            shuffle=True, 
                            num_workers=0, 
                            pin_memory=False, 
                            #persistent_workers=True
                            ) 
    
    val_loader = DataLoader(
                        val_ds, 
                        batch_size=1, 
                        shuffle=False, 
                        num_workers=0, 
                        pin_memory=False, 
                        #persistent_workers=True
                        )

    # Create generator model
    generator = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128),
                strides=(2, 2, 2),
                kernel_size=3).to(device)
    
    # Create Disciminator model
    discriminator = Discriminator(in_channels=1, features=[16, 32, 64, 128], kernel_size=[3,3,3]).to(device)

    # Init criterion
    BCE_LOSS = torch.nn.BCEWithLogitsLoss()
    L1_LOSS = torch.nn.L1Loss()
    torch.backends.cudnn.benchmark = True

    # Add optimizer
    d_lr = 0.00005 # discriminator learning rate
    g_lr = 0.0025 # generator learning rate
    optimizerG = optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # 🐝 Initialize wandb run
    wandb.init(project=f'cGAN-{in_contrast}2{out_contrast}', config=vars(args))

    # 🐝 Log gen gradients of the models to wandb
    wandb.watch(generator, log_freq=100)
    
    # 🐝 Add training script as an artifact
    artifact_script = wandb.Artifact(name='training', type='file')
    artifact_script.add_file(local_path=os.path.abspath(__file__), name=os.path.basename(__file__))
    wandb.log_artifact(artifact_script)

    # start a typical PyTorch training
    val_loss = np.inf
    for epoch in range(args.start_epoch, args.nb_epochs):
        print('\nEpoch: %d | GEN_LR: %.8f | DISC_LR: %.8f' % (epoch + 1, g_lr, d_lr))

        # train for one epoch
        train_Gloss, train_Dloss, train_Dacc = train(train_loader, generator, discriminator, BCE_LOSS, L1_LOSS, optimizerG, optimizerD, g_scaler, d_scaler, args.alpha, device)

        # 🐝 Plot G_loss D_loss and discriminator accuracy
        wandb.log({"Gloss_train/epoch": train_Gloss})
        wandb.log({"Dloss_train/epoch": train_Dloss})
        wandb.log({"Dacc_train/epoch": train_Dacc})
        
        # evaluate on validation set
        val_Gloss, val_Dloss, val_Dacc = validate(val_loader, generator, discriminator, BCE_LOSS, L1_LOSS, epoch, args.alpha, device)

        # 🐝 Plot G_loss D_loss and discriminator accuracy
        wandb.log({"Gloss_val/epoch": val_Gloss})
        wandb.log({"Dloss_val/epoch": val_Dloss})
        wandb.log({"Dacc_val/epoch": val_Dacc})
        
        # remember best acc and save checkpoint
        if val_loss > val_Gloss:
            val_loss = val_Gloss
            stateG = copy.deepcopy({'generator_weights': generator.state_dict()})
            torch.save(stateG, f'{weight_folder}/gen_{in_contrast}2{out_contrast}.pth')
            stateD = copy.deepcopy({'discriminator_weights': discriminator.state_dict()})
            torch.save(stateG, f'{weight_folder}/disc_{in_contrast}2{out_contrast}.pth')
    
    # 🐝 log best score and epoch number to wandb
    wandb.log({"best_accuracy": best_acc, "best_accuracy_epoch": best_acc_epoch})

    # 🐝 version your model
    best_model_path = f'{weight_folder}/gen_{in_contrast}2{out_contrast}.pth'
    model_artifact = wandb.Artifact(f"cGAN_{in_contrast}2{out_contrast}", 
                                    type="model",
                                    description=f"UNETR {in_contrast}2{out_contrast}",
                                    metadata=vars(args)
                                    )
    model_artifact.add_file(best_model_path)
    wandb.log_artifact(model_artifact)
        
    # 🐝 close wandb run
    wandb.finish()


def validate(data_loader, generator, discriminator, bce_loss, l1_loss, epoch, alpha, device):
    generator.eval()
    discriminator.eval()
    epoch_iterator = tqdm(data_loader, desc="Validation (G_loss=X.X) (D_loss=X.X) (ACC=X.X)", dynamic_ncols=True)
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            # Load input and target
            x, y = (batch["image"].to(device), batch["label"].to(device))

            # Get output from generator
            y_fake = generator(x)

            # Evaluate discriminator
            with torch.cuda.amp.autocast():
                D_real = discriminator(x, y)
                D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
                D_fake = discriminator(x, y_fake.detach())
                D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

            acc_real = D_real.mean().item() 
            acc_fake = 1.0 - D_fake.mean().item() 
            acc = (acc_real + acc_fake) / 2.0

            # Evaluate generator
            with torch.cuda.amp.autocast():
                D_fake = discriminator(x, y_fake)
                G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
                L1 = l1_loss(y_fake, y) * alpha
                G_loss = G_fake_loss + L1

            epoch_iterator.set_description(
                "Validation (G_loss=%2.5f) (D_loss=%2.5f) (ACC=%2.5f)" % (G_loss.mean().item(), D_loss.mean().item(), acc)
            )

    return G_loss.mean().item(), D_loss.mean().item(), acc


def train(data_loader, generator, discriminator, bce_loss, l1_loss, optimizerG, optimizerD, g_scaler, d_scaler, alpha, device):
    generator.train()
    discriminator.train()
    R, S, P = [], [], []
    epoch_iterator = tqdm(data_loader, desc="Training (G_loss=X.X) (D_loss=X.X) (ACC=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        # Load input and target
        x, y = (batch["image"].to(device), batch["label"].to(device))

        with torch.cuda.amp.autocast():
            # Get output from generator
            y_fake = generator(x)
            # Train discriminator
            D_real = discriminator(x, y)
            D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
            D_fake = discriminator(x, y_fake.detach())
            D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        discriminator.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(optimizerD)
        d_scaler.update()

        acc_real = D_real.mean().item() 
        acc_fake = 1.0 - D_fake.mean().item() 
        acc = (acc_real + acc_fake) / 2.0
        
        with torch.cuda.amp.autocast():
            # Train generator
            D_fake = discriminator(x, y_fake)
            G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * alpha
            G_loss = G_fake_loss + L1

        generator.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(optimizerG)
        g_scaler.update()

        epoch_iterator.set_description(
            "Training (G_loss=%2.5f) (D_loss=%2.5f)" % (G_loss.mean().item(), D_loss.mean().item())
        )

    return G_loss.mean().item(), D_loss.mean().item(), acc
    

if __name__=='__main__':
    main()