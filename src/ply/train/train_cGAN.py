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
from monai.networks.nets import UNet, AttentionUnet, SwinUNETR, UNETR
from monai.transforms import (
    LoadImaged,
    Orientationd,
    EnsureChannelFirstd,
    Spacingd,
    Compose,
    ResizeWithPadOrCropd,
    RandFlipd,
    NormalizeIntensityd
)

from ply.utils.utils import tuple_type, tuple2string
from ply.utils.config2parser import parser2config
from ply.train.utils import adjust_learning_rate
from ply.models.discriminator import Discriminator
from ply.models.criterion import CriterionCGAN
from ply.models.transform import RandLabelToContourd
from ply.utils.load_config import fetch_and_preproc_config_cGAN
from ply.utils.plot import get_validation_image


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train cGAN')
    parser.add_argument('--config', required=True, help='Config JSON file where every label used for TRAINING, VALIDATION and TESTING has its path specified ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--model', default='attunet', choices=['attunet', 'unetr', 'swinunetr'] , help='Model used for training. Options:["attunet", "unetr", "swinunetr"] (default="attunet")')
    parser.add_argument('--batch-size', type=int, default=3, help='Training batch size (default=3).')
    parser.add_argument('--nb-epochs', type=int, default=300, help='Number of training epochs (default=300).')
    parser.add_argument('--start-epoch', type=int, default=0, help='Starting epoch (default=0).')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Number of epochs during which the discriminator model will not learn (default=10).')
    parser.add_argument('--crop-size', type=tuple_type, default=(64, 320, 192), help='Training crop size in RSP orientation(default=(64, 320, 192)).')
    parser.add_argument('--channels', type=tuple_type, default=(16, 32, 64, 128, 256), help='Channels if attunet selected (default=16,32,64,128,256)')
    parser.add_argument('--pixdim', type=tuple_type, default=(0.8, 0.8, 0.8), help='Training resolution in RSP orientation (default=(0.8, 0.8, 0.8)).')
    parser.add_argument('--alpha', type=int, default=100, help='L1 loss multiplier (default=100).')
    parser.add_argument('--g-lr', default=2.5e-4, type=float, metavar='LR', help='Initial learning rate of the generator (default=2.5e-4)')
    parser.add_argument('--d-lr', default=2.5e-5, type=float, metavar='LR', help='Initial learning rate of the discriminator (default=2.5e-5)')
    parser.add_argument('--schedule', type=tuple_type, default=(0.50, 0.75), help='Decrease learning rate at these steps: fractions of the maximum number of epochs. (default=(0.5, 0.75))')
    parser.add_argument('--weight-folder', type=str, default=os.path.abspath('src/ply/weights/3D-CGAN'),
                        help='Folder where the cGAN weights will be stored and loaded. Will be created if does not exist. (default="src/ply/weights/3DGAN")')
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
    weight_folder = args.weight_folder
    in_contrast = config_data['CONTRASTS'].replace('_T2w','').replace('T2w_','')
    out_contrast = 'T2w'

    if len(out_contrast.split('_'))>1:
        raise ValueError(f'Multiple output contrast detected, check data config["CONTRAST"]={out_contrast}')
    
    # Save training config
    model = args.model if args.model != 'attunet' else f'{args.model}{str(args.channels[-1])}'
    json_name = f'config_cGAN_{model}_{in_contrast}2{out_contrast}_pixdimRSP_{tuple2string(args.pixdim)}_cropRSP_{tuple2string(args.crop_size)}_gLR_{str(args.g_lr)}_dLR_{str(args.d_lr)}.json'
    saved_args = copy.copy(args)
    parser2config(saved_args, path_out=os.path.join(weight_folder, json_name))  # Create json file with training parameters

    # Create weights folder to store training weights
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)
    
    # Load images for training and validation
    print('loading images...')
    train_list, err_train = fetch_and_preproc_config_cGAN(
                                            config_data=config_data,
                                            split='TRAINING'
                                            )
    
    val_list, err_val = fetch_and_preproc_config_cGAN(
                                            config_data=config_data,
                                            split='VALIDATION'
                                            )
    
    # Define transforms
    # Max with pixdim=(1, 1, 1)
    # R max =  51
    # S max =  234
    # P max =  156
    # Max with pixdim=(0.8, 0.8, 0.8)
    # R max = 64
    # S max = 292
    # P max = 195
    crop_size = args.crop_size # RSP
    pixdim = args.pixdim
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="LIA"), # RSP --> LIA
            Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "bilinear"),
            ),
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
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=crop_size,),
            RandLabelToContourd(keys=["image"], kernel_type='Laplace', prob=0.3),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            NormalizeIntensityd(keys=["label"], nonzero=False, channel_wise=False),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="LIA"), # RSP --> LIA
            Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "bilinear"),
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=crop_size,),
            RandLabelToContourd(keys=["image"], kernel_type='Laplace', prob=0.3),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            NormalizeIntensityd(keys=["label"], nonzero=False, channel_wise=False),
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
                            num_workers=4, 
                            pin_memory=True, 
                            persistent_workers=True
                            ) 
    
    val_loader = DataLoader(
                        val_ds, 
                        batch_size=args.batch_size, 
                        shuffle=False, 
                        num_workers=4, 
                        pin_memory=True, 
                        persistent_workers=True
                        )

    # Create generator model
    channels=args.channels
    if args.model == 'attunet':
        generator = AttentionUnet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=1,
                    channels=channels,
                    strides=[2]*(len(channels)-1),
                    kernel_size=3).to(device)
    elif args.model == 'swinunetr':
        generator =  SwinUNETR(
                        spatial_dims=3,
                        in_channels=1, 
                        out_channels=1, 
                        img_size=crop_size,
                        feature_size=24).to(device)
    elif args.model == 'unetr':
        generator = UNETR(
                        in_channels=1,
                        out_channels=1,
                        img_size=crop_size,
                        feature_size=16,
                        hidden_size=768,
                        mlp_dim=3072,
                        num_heads=12,
                        pos_embed="perceptron",
                        norm_name="instance",
                        res_block=True,
                        dropout_rate=0.0,
                    ).to(device)
    else:
        raise ValueError(f'Specified model {args.model} is unknown')

    # Create Disciminator model
    discriminator = Discriminator(in_channels=1, features=channels, kernel_size=[3,3,3]).to(device)

    # Init criterion
    BCE_LOSS = torch.nn.BCEWithLogitsLoss()
    FEATURE_LOSS = CriterionCGAN(dim=3, L1coeff=args.alpha, SSIMcoeff=100)
    torch.backends.cudnn.benchmark = True

    # Add optimizer
    d_lr = args.d_lr  # discriminator learning rate
    g_lr = args.g_lr  # generator learning rate
    optimizerG = optim.Adam(generator.parameters(), lr=g_lr, betas=(0.9, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.9, 0.999))
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
        # Adjust learning rate
        if epoch in [round(frac*args.nb_epochs) for frac in args.schedule]:
            g_lr = adjust_learning_rate(optimizerG, g_lr, gamma=0.5)
            d_lr = adjust_learning_rate(optimizerD, d_lr, gamma=0.8)

        print('\nEpoch: %d | GEN_LR: %.8f | DISC_LR: %.8f' % (epoch + 1, g_lr, d_lr))

        # train for one epoch
        warmup = True if epoch < args.warmup_epochs else False
        train_Gloss, train_Dloss, train_Dacc = train(train_loader, generator, discriminator, BCE_LOSS, FEATURE_LOSS, optimizerG, optimizerD, g_scaler, d_scaler, warmup, device)

        # 🐝 Plot G_loss D_loss and discriminator accuracy
        wandb.log({"Gloss_train/epoch": train_Gloss})
        wandb.log({"Dloss_train/epoch": train_Dloss})
        wandb.log({"Dacc_train/epoch": train_Dacc})
        wandb.log({"training_lr/epoch": g_lr})
        
        # evaluate on validation set
        val_Gloss, val_Dloss, val_Dacc = validate(val_loader, generator, discriminator, BCE_LOSS, FEATURE_LOSS, epoch, device)

        # 🐝 Plot G_loss D_loss and discriminator accuracy
        wandb.log({"Gloss_val/epoch": val_Gloss})
        wandb.log({"Dloss_val/epoch": val_Dloss})
        wandb.log({"Dacc_val/epoch": val_Dacc})
        
        # remember best acc and save checkpoint
        if val_loss > val_Gloss:
            val_loss = val_Gloss
            stateG = copy.deepcopy({'generator_weights': generator.state_dict()})
            torch.save(stateG, f'{weight_folder}/gen_{model}_{in_contrast}2{out_contrast}_alpha_{args.alpha}_pixdimRSP_{tuple2string(pixdim)}_cropRSP_{tuple2string(crop_size)}_gLR_{str(args.g_lr)}_dLR_{str(args.d_lr)}.pth')
            stateD = copy.deepcopy({'discriminator_weights': discriminator.state_dict()})
            torch.save(stateG, f'{weight_folder}/disc_{model}_{in_contrast}2{out_contrast}_alpha_{args.alpha}_pixdimRSP_{tuple2string(pixdim)}_cropRSP_{tuple2string(crop_size)}_gLR_{str(args.g_lr)}_dLR_{str(args.d_lr)}.pth')

    # 🐝 version your model
    best_model_path = f'{weight_folder}/gen_{model}_{in_contrast}2{out_contrast}_alpha_{args.alpha}_pixdimRSP_{tuple2string(pixdim)}_cropRSP_{tuple2string(crop_size)}_gLR_{str(args.g_lr)}_dLR_{str(args.d_lr)}.pth'
    model_artifact = wandb.Artifact(f"cGAN_{in_contrast}2{out_contrast}", 
                                    type="model",
                                    description=f"UNETR {in_contrast}2{out_contrast}",
                                    metadata=vars(args)
                                    )
    model_artifact.add_file(best_model_path)
    wandb.log_artifact(model_artifact)
        
    # 🐝 close wandb run
    wandb.finish()


def validate(data_loader, generator, discriminator, bce_loss, feature_loss, epoch, device):
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
                f_loss = feature_loss(y_fake, y)
                G_loss = G_fake_loss + f_loss

            epoch_iterator.set_description(
                "Validation (G_loss=%2.5f) (D_loss=%2.5f) (ACC=%2.5f)" % (G_loss.mean().item(), D_loss.mean().item(), acc)
            )

            # Display first image
            if step == 0:
                res_img, target_img, pred_img = get_validation_image(x, y, y_fake)

                # 🐝 log visuals for the first validation batch only in wandb
                wandb.log({"validation_img/batch_1": wandb.Image(res_img, caption=f'res_{epoch}')})
                wandb.log({"validation_img/groud_truth": wandb.Image(target_img, caption=f'ground_truth_{epoch}')})
                wandb.log({"validation_img/prediction": wandb.Image(pred_img, caption=f'prediction_{epoch}')})

    return G_loss.mean().item(), D_loss.mean().item(), acc


def train(data_loader, generator, discriminator, bce_loss, feature_loss, optimizerG, optimizerD, g_scaler, d_scaler, warmup, device):
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

        if not warmup:
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
            f_loss = feature_loss(y_fake, y)
            G_loss = G_fake_loss + f_loss

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