import logging
import os
import sys
import shutil
import tempfile
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
import torch

import monai
from monai.data import DataLoader, CacheDataset
from monai.networks.nets import UNETR
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
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
    parser.add_argument('--nb-epochs', type=int, default=200, help='Number of training epochs (default=200).')
    parser.add_argument('--start-epoch', type=int, default=0, help='Starting epoch (default=0).')
    parser.add_argument('--weight-folder', type=str, default=os.path.abspath('src/ply/weights/3D-CGAN'),
                        help='Folder where the cGAN weights will be stored and loaded. Will be created if does not exist. (default="src/ply/weights/3DGAN")')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

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

    # Create weights folder to store training weights
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)
    
    # Load config data
    # Read json file and create a dictionary
    with open(args.config_data, "r") as file:
        config_data = json.load(file)
    
    # Load variables
    weight_folder = args.weight_folder
    in_contrast = args.contrast
    out_contrast = config_data.CONTRASTS

    if len(out_contrast.split('_'))>1:
        raise ValueError(f'Multiple output contrast detected, check data config["CONTRAST"]={out_contrast}')
    
    # Load images for training and validation
    print('loading images...')
    train_list = fetch_and_preproc_config_cGAN(
                                            config_data=config_data,
                                            cont=args.contrast,
                                            split='TRAINING'
                                            )
    
    val_list = fetch_and_preproc_config_cGAN(
                                            config_data=config_data,
                                            cont=args.contrast,
                                            split='VALIDATION'
                                            )
    
    # Define transforms
    crop_size = (192, 168, 60)
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
            #ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=crop_size,), # TODO
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
            #ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=crop_size,), # TODO,
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
                            num_workers=16, 
                            pin_memory=True, 
                            persistent_workers=True
                            ) 
    
    val_loader = DataLoader(
                        val_ds, 
                        batch_size=1, 
                        shuffle=False, 
                        num_workers=16, 
                        pin_memory=True, 
                        persistent_workers=True
                        )

    # Create generator model
    generator = UNETR(
            in_channels=1,
            out_channels=1,
            img_size=(96, 96, 96), # TODO
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0).to(device)
    
    # Create Disciminator model
    discriminator = Disciminator(feature_size=32, kernel_size=[3,3,3]).to(device)

    # Init criterion
    BCE_LOSS = torch.nn.BCELoss()
    L1_LOSS = torch.nn.L1Loss()
    torch.backends.cudnn.benchmark = True

    # Add optimizer
    optimizerG = optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))

    # 🐝 Initialize wandb run
    wandb.init(project=f'cGAN-{in_contrast}2{out_contrast}', config=vars(args))

    # 🐝 Log gen gradients of the models to wandb
    wandb.watch(generator, log_freq=100)
    
    # 🐝 Add training script as an artifact
    artifact_script = wandb.Artifact(name='training', type='file')
    artifact_script.add_file(local_path=os.path.abspath(__file__), name=os.path.basename(__file__))
    wandb.log_artifact(artifact_script)

    # start a typical PyTorch training
    gen_loss = np.inf
    for epoch in range(args.start_epoch, args.epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # train for one epoch
        train_Gloss, train_Dloss, train_Dacc = train(train_loader, generator, discriminator, BCE_LOSS, L1_LOSS, optimizerG, optimizerD, device)

        if wandb_mode:
            # 🐝 Plot G_loss
            wandb.log({"Gloss_train/epoch": train_Gloss})

            # 🐝 Plot D_loss
            wandb.log({"Dloss_train/epoch": train_Dloss})
            
            # 🐝 Plot discriminator accuracy
            wandb.log({"Dacc_train/epoch": train_Dacc})
        
        # evaluate on validation set
        val_Gloss, val_Dloss, val_Dacc = validate(val_loader, generator, discriminator, BCE_LOSS, L1_LOSS, device)

        if wandb_mode:
            # 🐝 Plot G_loss
            wandb.log({"Gloss_val/epoch": val_Gloss})

            # 🐝 Plot D_loss
            wandb.log({"Dloss_val/epoch": val_Dloss})
            
            # 🐝 Plot discriminator accuracy
            wandb.log({"Dacc_val/epoch": val_Dacc})
        
        # remember best acc and save checkpoint
        if gen_loss > val_Gloss:
            gen_loss = val_Gloss
            stateG = copy.deepcopy({'generator_weights': generator.state_dict()})
            torch.save(stateG, f'{weight_folder}/gen_{in_contrast}2{out_contrast}.pth')
            stateD = copy.deepcopy({'discriminator_weights': discriminator.state_dict()})
            torch.save(stateG, f'{weight_folder}/disc_{in_contrast}2{out_contrast}.pth')
    
    if wandb_mode:
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


def validate(data_loader, generator, discriminator, bce_loss, l1_loss, epoch, device):
    model.eval()
    epoch_iterator = tqdm(data_loader, desc="Validation (X / X Steps) (G_loss=X.X) (D_loss=X.X) (ACC=X.X)", dynamic_ncols=True)
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            # Load input and target
            x, y = (batch["image"].to(device), batch["label"].to(device))

            # Get output from generator
            y_fake = generator(x)

            # Evaluate discriminator
            D_real = discriminator(x, y)
            D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
            D_fake = discriminator(x, y_fake.detach())
            D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

            acc_real = D_real.mean().item() 
            acc_fake = 1.0 - D_fake.mean().item() 
            acc = (acc_real + acc_fake) / 2.0

            # Evaluate generator
            # D_fake = discriminator(x, y_fake)
            G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * 1
            G_loss = G_fake_loss + L1

            epoch_iterator.set_description(
                "Validation (%d / %d Steps ) (G_loss=%2.5f) (D_loss=%2.5f) (ACC=%2.5f)" % (step, len(data_loader), G_loss.mean().item(), D_loss.mean().item(), acc)
            )

    return G_loss.mean().item(), D_loss.mean().item(), acc


def train(data_loader, generator, discriminator, bce_loss, l1_loss, optimizerG, optimizerD, epoch, device):
    model.train()
    epoch_iterator = tqdm(data_loader, desc="Training (X / X Steps) (G_loss=X.X) (D_loss=X.X) (ACC=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        # Load input and target
        x, y = (batch["image"].to(device), batch["label"].to(device))

        # # Get output from generator
        # y_fake = generator(x)
        print(1)
        # # Train discriminator
        # D_real = discriminator(x, y)
        # D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
        # D_fake = discriminator(x, y_fake.detach())
        # D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
        # D_loss = (D_real_loss + D_fake_loss) / 2

        # discriminator.zero_grad()
        # optimizerD.scale(D_loss).backward()
        # optimizerD.step()
        # optimizerD.update()

        # acc_real = D_real.mean().item() 
        # acc_fake = 1.0 - D_fake.mean().item() 
        # acc = (acc_real + acc_fake) / 2.0
        
        # # Train generator
        # # D_fake = discriminator(x, y_fake)
        # G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
        # L1 = l1_loss(y_fake, y) * 1
        # G_loss = G_fake_loss + L1

        # generator.zero_grad()
        # optimizerG.scale(G_loss).backward()
        # optimizerG.step()
        # optimizerG.update()

        # epoch_iterator.set_description(
        #     "Training (%d / %d Steps ) (G_loss=%2.5f) (D_loss=%2.5f)" % (step, len(data_loader), G_loss.mean().item(), D_loss.mean().item())
        # )

    return G_loss.mean().item(), D_loss.mean().item(), acc
    

if __name__=='__main__':
    main()