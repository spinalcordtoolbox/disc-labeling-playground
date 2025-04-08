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
import torch.nn.functional as F

import monai
from monai.data import DataLoader, CacheDataset
from monai.networks.nets import UNet, SwinUNETR, UNETR
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss
from monai.transforms import (
    LoadImaged,
    Orientationd,
    EnsureChannelFirstd,
    Spacingd,
    Compose,
    RandFlipd,
    NormalizeIntensityd,
    RandSpatialCropSamplesd,
    ResizeWithPadOrCropd
)

from ply.utils.utils import tuple_type_int, tuple_type_float, tuple2string, normalize, qc_reg_rgb, qc_side_by_side, compute_dsc
from ply.utils.config2parser import parser2config
from ply.train.utils import adjust_learning_rate
from ply.models.transform import RandLabelToContourd, ConvertDsegToMultiChannels
from ply.utils.load_config import fetch_data_config
from ply.utils.plot import get_validation_image
from ply.models.segmentation.attunet import AttentionUnet


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train segmentation model for vertebrae')
    parser.add_argument('--config', required=True, help='Config JSON file where every label used for TRAINING, VALIDATION and TESTING has its path specified ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--model', type=str, default='attunet', choices=['attunet', 'unetr', 'swinunetr'] , help='Model used for training. Options:["attunet", "unetr", "swinunetr"] (default="attunet")')
    parser.add_argument('--batch-size', type=int, default=1, help='Training batch size (default=1).')
    parser.add_argument('--nb-epochs', type=int, default=1000, help='Number of training epochs (default=1000).')
    parser.add_argument('--start-epoch', type=int, default=0, help='Starting epoch (default=0).')
    parser.add_argument('--schedule', type=tuple_type_float, default=tuple([0.3, 0.6, 0.9]), help='Fraction of the max epoch where the learning rate will be reduced of a factor gamma (default=(0.3, 0.6, 0.9)).')
    parser.add_argument('--gamma', type=float, default=0.1, help='Factor used to reduce the learning rate (default=0.1)')
    parser.add_argument('--crop-size', type=tuple_type_int, default=(96, 96, 96), help='Training crop size in RSP orientation(default=(64, 64, 64)).')
    parser.add_argument('--channels', type=tuple_type_int, default=(16, 32, 64, 128, 256, 512), help='Channels if attunet selected (default=16, 32,64,128,256, 512)')
    parser.add_argument('--pixdim', type=tuple_type_float, default=(1, 1, 1), help='Training resolution in RSP orientation (default=(1, 1, 1)).')
    parser.add_argument('--lr', default=1e-4, type=float, metavar='LR', help='Initial learning rate (default=1e-4)')
    parser.add_argument('--weight-folder', type=str, default=os.path.abspath('src/ply/weights/3DSegVert'), help='Folder where the weights will be stored and loaded. Will be created if does not exist. (default="src/ply/weights/3DSegVert")')
    parser.add_argument('--start-weights', type=str, default='', help='Path to the model weights used to start the training.')
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

    # Create weights folder to store training weights
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)
    
    # Save training config
    model_name = args.model if args.model != 'attunet' else f'{args.model}{str(args.channels[-1])}'
    json_name = f'config_SegVert_{model_name}_pixdimRSP_{tuple2string(args.pixdim)}_cropRSP_{tuple2string(args.crop_size)}_LR_{str(args.lr)}.json'
    saved_args = copy.copy(args)
    parser2config(saved_args, path_out=os.path.join(weight_folder, json_name))  # Create json file with training parameters
    
    # Load images for training and validation
    print('loading images...')
    train_list, err_train = fetch_data_config(
        config_data=config_data,
        split='TRAINING'
    )
    
    val_list, err_val = fetch_data_config(
        config_data=config_data,
        split='VALIDATION'
    )
    
    # Define transforms
    crop_size = args.crop_size # RSP
    pixdim = args.pixdim
    
    # Train and val transforms
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image"]),
            ConvertDsegToMultiChannels(keys=["label"]), # Add labels to different channels
            Orientationd(keys=["image", "label"], axcodes="LIA"), # RSP --> LIA
            Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=(2, "nearest"), # 2 for spline interpolation
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
            RandSpatialCropSamplesd(keys=["image", "label"], roi_size=crop_size, num_samples=12, random_size=False),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=crop_size),
            RandLabelToContourd(keys=["image"], kernel_type="Scharr", prob=0.2),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image"]),
            ConvertDsegToMultiChannels(keys=["label"]),
            Orientationd(keys=["image", "label"], axcodes="LIA"), # RSP --> LIA
            Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=(2, "nearest"),
            ),
            RandSpatialCropSamplesd(keys=["image", "label"], roi_size=crop_size, num_samples=4, random_size=False),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=crop_size),
            RandLabelToContourd(keys=["image"], kernel_type="Scharr", prob=0.2),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ]
    )

    # Define train and val dataset
    train_ds = CacheDataset(
                            data=train_list,
                            transform=train_transforms,
                            cache_rate=0,
                            )
    val_ds = CacheDataset(
                        data=val_list,
                        transform=val_transforms,
                        cache_rate=0,
                        )

    # Define train and val DataLoader
    train_loader = DataLoader(
                            train_ds, 
                            batch_size=args.batch_size,
                            shuffle=True, 
                            num_workers=5, 
                            pin_memory=False, 
                            persistent_workers=False
                            ) 
    
    val_loader = DataLoader(
                        val_ds, 
                        batch_size=args.batch_size, 
                        shuffle=False, 
                        num_workers=5, 
                        pin_memory=False, 
                        persistent_workers=False
                        )

    # Create model
    channels=args.channels
    out_channels = 1
    if args.model == 'attunet':
        model = AttentionUnet(
                    spatial_dims=3,
                    in_shape=crop_size,
                    in_channels=1,
                    out_channels=out_channels,
                    channels=channels,
                    strides=[2]*(len(channels)-1),
                    kernel_size=3).to(device)
    elif args.model == 'swinunetr':
        model =  SwinUNETR(
                        spatial_dims=3,
                        in_channels=1, 
                        out_channels=out_channels, 
                        img_size=crop_size,
                        feature_size=24).to(device)
    elif args.model == 'unetr':
        model = UNETR(
                        in_channels=1,
                        out_channels=out_channels,
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

    # Init weights if weights are specified
    if args.start_weights:
        # Check if weights path exists
        if not os.path.exists(args.start_weights):
            raise ValueError(f'Weights path {args.start_weights} does not exist')
        else:
            # Load model weights
            model.load_state_dict(torch.load(args.start_weights, map_location=torch.device(device))["weights"])

    # Path to the saved weights       
    weights_path = f'{weight_folder}/{json_name.replace("config_SegVert_","").replace(".json", ".pth")}'

    # Init criterion
    loss_func = DiceFocalLoss(sigmoid=True, smooth_dr=1e-4)
    torch.backends.cudnn.benchmark = True

    # Add optimizer
    lr = args.lr  # learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    scaler = torch.cuda.amp.GradScaler()

    # 🐝 Initialize wandb run
    wandb.init(project=f'Vertebrae-Segmentation', config=vars(args))

    # 🐝 Log gen gradients of the models to wandb
    wandb.watch(model, log_freq=100)
    
    # 🐝 Add training script as an artifact
    artifact_script = wandb.Artifact(name='training', type='file')
    artifact_script.add_file(local_path=os.path.abspath(__file__), name=os.path.basename(__file__))
    wandb.log_artifact(artifact_script)

    # start a typical PyTorch training
    val_dsc_best = 0
    for epoch in range(args.start_epoch, args.nb_epochs):
        # Adjust learning rate
        if epoch in [int(sch*args.nb_epochs) for sch in args.schedule]:
            lr = adjust_learning_rate(optimizer, lr, gamma=args.gamma)

        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # train for one epoch
        train_loss, train_dsc = train(train_loader, model, loss_func, optimizer, scaler, device)

        # 🐝 Plot loss and dice similarity coefficient
        wandb.log({"Loss_train/epoch": train_loss})
        wandb.log({"DSC_train/epoch": train_dsc})
        wandb.log({"training_lr/epoch": lr})
        
        # evaluate on validation set
        val_loss, val_dsc = validate(val_loader, model, loss_func, epoch, device)

        # 🐝 Plot loss and dice similarity coefficient
        wandb.log({"Loss_val/epoch": val_loss})
        wandb.log({"DSC_val/epoch": val_dsc})
        
        # remember best acc and save checkpoint
        if val_dsc > val_dsc_best:
            val_dsc_best = val_dsc
            state = copy.deepcopy({'weights': model.state_dict()})
            torch.save(state, weights_path)
        
    # 🐝 close wandb run
    wandb.finish()


def validate(data_loader, model, loss_func, epoch, device):
    model.eval()
    dsc_list = [0]
    epoch_iterator = tqdm(data_loader, desc="Validation (loss=X.X) (DSC=X.X)", dynamic_ncols=True)
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            # Load input and target
            x, y = (batch["image"].to(device), batch["label"].to(device))

            # Get output from model
            y_pred = model(x)

            # Compute loss for each element in the batch size
            loss = 0
            for i in range(y_pred.shape[0]):
                y1 = y[i, 0].unsqueeze(0).detach().clone().to(device)
                y2 = y[i, 1].unsqueeze(0).detach().clone().to(device)
                loss1 = loss_func(y_pred[i], y1)
                loss2 = loss_func(y_pred[i], y2)
                loss += min(loss1, loss2)

                # Calculate DSC
                dsc1 = compute_dsc(y1.detach().cpu().numpy(), y_pred[i].detach().cpu().numpy(), sigmoid=True)
                dsc2 = compute_dsc(y2.detach().cpu().numpy(), y_pred[i].detach().cpu().numpy(), sigmoid=True)
                if dsc1 > 0 or dsc2 > 0:
                    dsc_list.append(max(dsc1, dsc2))

            epoch_iterator.set_description(
                "Validation (loss=%2.5f) (DSC=%2.5f)" % (loss.mean().item(), np.mean(dsc_list))
            )

            # Display first image
            if step == 0:
                res_img, target_img, pred_img = get_validation_image(x, y, y_pred)

                # 🐝 log visuals for the first validation batch only in wandb
                wandb.log({"validation_img/batch_1": wandb.Image(res_img, caption=f'res_{epoch}')})
                wandb.log({"validation_img/groud_truth": wandb.Image(target_img, caption=f'ground_truth_{epoch}')})
                wandb.log({"validation_img/prediction": wandb.Image(pred_img, caption=f'prediction_{epoch}')})

    return loss.mean().item(), np.mean(dsc_list)


def train(data_loader, model, loss_func, optimizer, scaler, device):
    model.train()
    dsc_list = [0]
    epoch_iterator = tqdm(data_loader, desc="Training (loss=X.X) (DSC=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        # Load input and target
        x, y = batch["image"].to(device), batch["label"].to(device)
        
        #qc_side_by_side(image_name=os.path.basename(x.meta['filename_or_obj'][0]), image=x.data.cpu().numpy()[0,0], target=y.data.cpu().numpy()[0,0], qc_path='./qc')
        #qc_reg_rgb(image_name=os.path.basename(x.meta['filename_or_obj'][0]), image=x.data.cpu().numpy()[0,0], target=y.data.cpu().numpy()[0,0], qc_path='./qc-rgb')
        with torch.amp.autocast('cuda'):
            # Get output from model
            y_pred = model(x)
            
            # Compute loss for each element in the batch size
            loss = 0
            for i in range(y_pred.shape[0]):
                y1 = y[i, 0].unsqueeze(0).detach().clone().to(device)
                y2 = y[i, 1].unsqueeze(0).detach().clone().to(device)
                loss1 = loss_func(y_pred[i], y1)
                loss2 = loss_func(y_pred[i], y2)
                loss += min(loss1, loss2)

                # Calculate DSC
                dsc1 = compute_dsc(y1.detach().cpu().numpy(), y_pred[i].detach().cpu().numpy(), sigmoid=True)
                dsc2 = compute_dsc(y2.detach().cpu().numpy(), y_pred[i].detach().cpu().numpy(), sigmoid=True)
                if dsc1 > 0 or dsc2 > 0:
                    dsc_list.append(max(dsc1, dsc2))

        # Train model
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_iterator.set_description(
            "Training (loss=%2.5f) (DSC=%2.5f)" % (loss.mean().item(), np.mean(dsc_list))
        )
    return loss.mean().item(), np.mean(dsc_list)
    

if __name__=='__main__':
    main()
