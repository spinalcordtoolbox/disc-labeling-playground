"""
Train a conditional WGAN-GP to translate any contrasts to T2w contrasts.

Based on:
- https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/3.%20WGAN/train.py
- https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/Pix2Pix/train.py
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

import torch
import torch.optim as optim

import monai
from monai.data import DataLoader, CacheDataset
from monai.networks.nets import AttentionUnet, SwinUNETR, UNETR
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

from ply.utils.utils import tuple_type_int, tuple_type_float, tuple2string, normalize
from ply.utils.config2parser import parser2config
from ply.train.utils import adjust_learning_rate
from ply.models.critic import Critic
from ply.models.criterion import CriterionCGAN, gradient_penalty
from ply.models.transform import RandLabelToContourd
from ply.utils.load_config import fetch_and_preproc_config_cGAN
from ply.utils.plot import get_validation_image


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train cWGAN-GP')
    parser.add_argument('--config', required=True, help='Config JSON file where every label used for TRAINING, VALIDATION and TESTING has its path specified ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--model', type=str, default='attunet', choices=['attunet', 'unetr', 'swinunetr'] , help='Model used for training. Options:["attunet", "unetr", "swinunetr"] (default="attunet")')
    parser.add_argument('--batch-size', type=int, default=3, help='Training batch size (default=3).')
    parser.add_argument('--nb-epochs', type=int, default=300, help='Number of training epochs (default=300).')
    parser.add_argument('--start-epoch', type=int, default=0, help='Starting epoch (default=0).')
    parser.add_argument('--schedule', type=tuple_type_float, default=tuple([(i+1)*0.3 for i in range(3)]), help='Fraction of the max epoch where the learning rate will be reduced of a factor gamma (default=(0.3, 0.6, 0.9)).')
    parser.add_argument('--gamma', type=float, default=0.5, help='Factor used to reduce the learning rate (default=0.5)')
    parser.add_argument('--crop-size', type=tuple_type_int, default=(64, 256, 192), help='Training crop size in RSP orientation(default=(64, 256, 192)).')
    parser.add_argument('--channels', type=tuple_type_int, default=(16, 32, 64, 128, 256), help='Channels if attunet selected (default=16,32,64,128,256)')
    parser.add_argument('--pixdim', type=tuple_type_float, default=(1, 1, 1), help='Training resolution in RSP orientation (default=(1, 1, 1)).')
    parser.add_argument('--laplace-prob', type=float, default=1, help='Probability to apply laplacian kernel to input for training. (default=1).')
    parser.add_argument('--interp-mode', type=str, default='spline', choices=['bilinear', 'nearest','spline'], help='Interpolation mode for input and output image. (default="spline").')
    parser.add_argument('--alpha', type=int, default=100, help='L1 loss multiplier (default=100).')
    parser.add_argument('--crit-iter', type=int, default=5, help='Training iteration of the critic when the generator is trained 1 time (default=5).')
    parser.add_argument('--lambda-gp', type=int, default=10, help='Gradient penalty hyperparameter (default=10).')
    parser.add_argument('--g-lr', default=1e-4, type=float, metavar='LR', help='Initial learning rate of the generator (default=1e-4)')
    parser.add_argument('--c-lr', default=1e-4, type=float, metavar='LR', help='Initial learning rate of the critic (default=1e-4)')
    parser.add_argument('--weight-folder', type=str, default=os.path.abspath('src/ply/weights/3D-CWGAN'), help='Folder where the cWGAN weights will be stored and loaded. Will be created if does not exist. (default="src/ply/weights/3D-CWGAN")')
    parser.add_argument('--start-gen-weights', type=str, default='', help='Path to the generator weights used to start the training.')
    parser.add_argument('--start-critic-weights', type=str, default='', help='Path to the critic weights used to start the training.')
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
    critic_iter = args.crit_iter
    lambda_gp = args.lambda_gp
    weight_folder = args.weight_folder
    in_contrast = config_data['CONTRASTS'].replace('_T2w','').replace('T2w_','')
    out_contrast = 'T2w'

    if len(out_contrast.split('_'))>1:
        raise ValueError(f'Multiple output contrast detected, check data config["CONTRAST"]={out_contrast}')
    
    # Save training config
    model = args.model if args.model != 'attunet' else f'{args.model}{str(args.channels[-1])}'
    json_name = f'config_cWGAN_{model}_{in_contrast}2{out_contrast}_laplace_{str(args.laplace_prob)}_pixdimRSP_{tuple2string(args.pixdim)}_cropRSP_{tuple2string(args.crop_size)}_gLR_{str(args.g_lr)}_cLR_{str(args.c_lr)}_gamma_{str(args.gamma)}_critIter_{critic_iter}.json'
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
    # Max with pixdim=(0.7, 0.7, 0.7)
    # R max = 128
    # S max = 332
    # P max = 205
    if args.interp_mode != 'spline':
        interp_mode = args.interp_mode
    else:
        interp_mode = 2
    crop_size = args.crop_size # RSP
    pixdim = args.pixdim
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="LIA"), # RSP- --> LIA+
            Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=(interp_mode, interp_mode),
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
            RandLabelToContourd(keys=["image"], kernel_type='Laplace', prob=args.laplace_prob),
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
                mode=(interp_mode, interp_mode),
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=crop_size,),
            RandLabelToContourd(keys=["image"], kernel_type='Laplace', prob=args.laplace_prob),
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

    # Create Critic model
    critic = Critic(in_channels=1, features=channels, kernel_size=[3,3,3]).to(device)

    # Init weights if weights are specified
    if args.start_gen_weights or args.start_critic_weights:
        if args.start_gen_weights:
            # Check if weights path exists
            if not os.path.exists(args.start_gen_weights):
                raise ValueError(f'Weights path {args.start_gen_weights} does not exist')
        # Load generator weights
        generator.load_state_dict(torch.load(args.start_gen_weights, map_location=torch.device(device))["generator_weights"])

        if args.start_critic_weights:
            # Check if weights path exists
            if not os.path.exists(args.start_critic_weights):
                raise ValueError(f'Weights path {args.start_critic_weights} does not exist')
        # Load critic weights
        critic.load_state_dict(torch.load(args.start_critic_weights, map_location=torch.device(device))["critic_weights"])

    # Path to the saved weights       
    gen_weights_path = f'{weight_folder}/gen_cWGAN_{model}_{in_contrast}2{out_contrast}_laplace_{str(args.laplace_prob)}_alpha_{args.alpha}_pixdimRSP_{tuple2string(pixdim)}_cropRSP_{tuple2string(crop_size)}_gLR_{str(args.g_lr)}_cLR_{str(args.c_lr)}_gamma_{str(args.gamma)}_critIter_{critic_iter}.pth'
    crit_weights_path = gen_weights_path.replace('gen', 'crit')

    # Init criterion
    FEATURE_LOSS = CriterionCGAN(dim=3, L1coeff=args.alpha, SSIMcoeff=100)
    torch.backends.cudnn.benchmark = True

    # Add optimizer
    c_lr = args.c_lr  # critic learning rate
    g_lr = args.g_lr  # generator learning rate
    optimizerG = optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
    optimizerC = optim.Adam(critic.parameters(), lr=c_lr, betas=(0.5, 0.999))
    g_scaler = torch.cuda.amp.GradScaler()
    c_scaler = torch.cuda.amp.GradScaler()

    # 🐝 Initialize wandb run
    wandb.init(project=f'cWGAN-{in_contrast}2{out_contrast}', config=vars(args))

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
        if epoch in [int(sch*args.nb_epochs) for sch in args.schedule]:
            g_lr = adjust_learning_rate(optimizerG, g_lr, gamma=args.gamma)
            c_lr = adjust_learning_rate(optimizerC, c_lr, gamma=args.gamma)

        print('\nEpoch: %d | GEN_LR: %.8f | CRIT_LR: %.8f' % (epoch + 1, g_lr, c_lr))

        # train for one epoch
        train_Gloss, train_Closs, train_Cacc = train(train_loader, generator, critic, critic_iter, lambda_gp, FEATURE_LOSS, optimizerG, optimizerC, g_scaler, c_scaler, device)

        # 🐝 Plot G_loss D_loss and critic accuracy
        wandb.log({"Gloss_train/epoch": train_Gloss})
        wandb.log({"Closs_train/epoch": train_Closs})
        wandb.log({"Cacc_train/epoch": train_Cacc})
        wandb.log({"training_lr/epoch": g_lr})
        
        # evaluate on validation set
        val_Gloss, val_Closs, val_Cacc = validate(val_loader, generator, critic, lambda_gp, FEATURE_LOSS, epoch, device)

        # 🐝 Plot G_loss D_loss and critic accuracy
        wandb.log({"Gloss_val/epoch": val_Gloss})
        wandb.log({"Closs_val/epoch": val_Closs})
        wandb.log({"Cacc_val/epoch": val_Cacc})
        
        # remember best acc and save checkpoint
        if val_loss > val_Gloss:
            val_loss = val_Gloss
            stateG = copy.deepcopy({'generator_weights': generator.state_dict()})
            torch.save(stateG, gen_weights_path)
            stateD = copy.deepcopy({'critic_weights': critic.state_dict()})
            torch.save(stateG, crit_weights_path)

    # 🐝 version your model
    best_model_path = gen_weights_path
    model_artifact = wandb.Artifact(f"cWGAN_{in_contrast}2{out_contrast}", 
                                    type="model",
                                    description=f"{args.model}_{in_contrast}2{out_contrast}",
                                    metadata=vars(args)
                                    )
    model_artifact.add_file(best_model_path)
    wandb.log_artifact(model_artifact)
        
    # 🐝 close wandb run
    wandb.finish()


def validate(data_loader, generator, critic, lambda_gp, feature_loss, epoch, device):
    generator.eval()
    critic.eval()
    acc_list = []
    epoch_iterator = tqdm(data_loader, desc="Validation (G_loss=X.X) (C_loss=X.X) (ACC=X.X)", dynamic_ncols=True)
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            # Load input and target
            x, y = (batch["image"].to(device), batch["label"].to(device))

            # Get output from generator
            y_fake = generator(x)

            # Evaluate critic
            with torch.cuda.amp.autocast():
                C_real = critic(x, y).reshape(-1)
                C_fake = critic(x, y_fake.detach()).reshape(-1)
                C_loss = -(torch.mean(C_real) - torch.mean(C_fake))

            acc_real = C_real.mean().item() 
            acc_fake = - C_fake.mean().item() 
            acc = (acc_real + acc_fake) / 2.0
            acc_list.append(acc)

            # Evaluate generator
            with torch.cuda.amp.autocast():
                C_fake = critic(x, y_fake).reshape(-1)
                G_fake_loss = -torch.mean(C_fake)
                f_loss = feature_loss(y_fake, y)
                G_loss = G_fake_loss + f_loss

            epoch_iterator.set_description(
                "Validation (G_loss=%2.5f) (C_loss=%2.5f) (ACC=%2.5f)" % (G_loss.mean().item(), C_loss.mean().item(), acc)
            )

            # Display first image
            if step == 0:
                res_img, target_img, pred_img = get_validation_image(x, y, y_fake)

                # 🐝 log visuals for the first validation batch only in wandb
                wandb.log({"validation_img/batch_1": wandb.Image(res_img, caption=f'res_{epoch}')})
                wandb.log({"validation_img/groud_truth": wandb.Image(target_img, caption=f'ground_truth_{epoch}')})
                wandb.log({"validation_img/prediction": wandb.Image(pred_img, caption=f'prediction_{epoch}')})

    return G_loss.mean().item(), C_loss.mean().item(), np.mean(acc_list)


def train(data_loader, generator, critic, critic_iter, lambda_gp, feature_loss, optimizerG, optimizerC, g_scaler, c_scaler, device):
    generator.train()
    critic.train()
    R, S, P = [], [], []
    acc_list = []
    epoch_iterator = tqdm(data_loader, desc="Training (G_loss=X.X) (C_loss=X.X) (ACC=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        # Load input and target
        x, y = (batch["image"].to(device), batch["label"].to(device))

        with torch.cuda.amp.autocast():
            # Get output from generator
            y_fake = generator(x)
            # Train critic
            for i in range(critic_iter):
                C_real = critic(x, y).reshape(-1)
                C_fake = critic(x, y_fake.detach()).reshape(-1)
                gp = gradient_penalty(critic, x, y, y_fake, device=device)
                C_loss = -(torch.mean(C_real) - torch.mean(C_fake)) + lambda_gp * gp
                critic.zero_grad()
                c_scaler.scale(C_loss).backward(retain_graph=True)
                c_scaler.step(optimizerC)
                c_scaler.update()

        acc_real = C_real.mean().item() 
        acc_fake = - C_fake.mean().item() 
        acc = (acc_real + acc_fake) / 2.0
        acc_list.append(acc)

        with torch.cuda.amp.autocast():
            # Train generator
            C_fake = critic(x, y_fake).reshape(-1)
            G_fake_loss = -torch.mean(C_fake)
            f_loss = feature_loss(y_fake, y)
            G_loss = G_fake_loss + f_loss

        generator.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(optimizerG)
        g_scaler.update()

        epoch_iterator.set_description(
            "Training (G_loss=%2.5f) (C_loss=%2.5f)" % (G_loss.mean().item(), C_loss.mean().item())
        )

    return G_loss.mean().item(), C_loss.mean().item(), np.mean(acc_list)
    

if __name__=='__main__':
    main()