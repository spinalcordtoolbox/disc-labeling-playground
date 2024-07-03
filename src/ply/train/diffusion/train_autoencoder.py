# Script based on https://github.com/Project-MONAI/tutorials/blob/main/generative/2d_ldm/train_autoencoder.py

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import wandb
import numpy as np
from tqdm import tqdm

import torch
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator
from monai.config import print_config
from monai.utils import set_determinism
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import define_instance, setup_ddp, prepare_dataloader

from ply.utils.plot import get_validation_image_diff_2d
from ply.models.diffusion.vqvae import VQVAE


def main():
    parser = argparse.ArgumentParser(description="PyTorch autoencoder training")
    parser.add_argument(
        "-e",
        "--environment-file",
        required=True,
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        required=True,
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()

    # Step 0: configuration
    ddp_bool = args.gpus > 1  # whether to use distributed data parallel
    if ddp_bool:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist, device = setup_ddp(rank, world_size)
    else:
        rank = 0
        world_size = 1
        device = 0

    torch.cuda.set_device(device)
    print(f"Using {device}")

    print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    torch.autograd.set_detect_anomaly(True)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    set_determinism(42)

    # Step 1: set data loader
    size_divisible = 2 ** (len(args.autoencoder_def["num_channels"]) - 1)
    train_loader, val_loader = prepare_dataloader(
        args,
        args.autoencoder_train["batch_size"],
        args.autoencoder_train["train_patch_size"],
        args.autoencoder_train["val_patch_size"],
        sample_axis=args.sample_axis,
        randcrop=True,
        rank=rank,
        world_size=world_size,
        cache=1.0,
        download=False,
        size_divisible=size_divisible,
        train_transform='crop',
        val_transform='full',
    )

    # Step 2: Define VQVAE network and discriminator
    autoencoder = VQVAE(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=args.autoencoder_def["num_channels"],
        num_res_channels=args.autoencoder_def["num_channels"][-1],
        num_res_layers=args.autoencoder_def["num_res_blocks"],
        num_embeddings=args.autoencoder_def["num_channels"][0],
        embedding_dim= 64
    ).to(device)

    discriminator_norm = "INSTANCE"
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=len(args.autoencoder_def["num_channels"]),
        num_channels=args.autoencoder_def["num_channels"][0]//2,
        in_channels=1,
        out_channels=1,
        norm=discriminator_norm,
    ).to(device)
    if ddp_bool:
        # When using DDP, BatchNorm needs to be converted to SyncBatchNorm.
        discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    trained_g_path = os.path.join(args.model_dir, "vqvae.pt")
    trained_d_path = os.path.join(args.model_dir, "discriminator.pt")
    trained_g_path_last = os.path.join(args.model_dir, "vqvae_last.pt")
    trained_d_path_last = os.path.join(args.model_dir, "discriminator_last.pt")

    if rank == 0:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    if args.resume_ckpt:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        try:
            autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location))
            print(f"Rank {rank}: Load trained autoencoder from {trained_g_path}")
        except:
            print(f"Rank {rank}: Train autoencoder from scratch.")

        try:
            discriminator.load_state_dict(torch.load(trained_d_path, map_location=map_location))
            print(f"Rank {rank}: Load trained discriminator from {trained_d_path}")
        except:
            print(f"Rank {rank}: Train discriminator from scratch.")

    if ddp_bool:
        autoencoder = DDP(
            autoencoder,
            device_ids=[device],
            output_device=rank,
            find_unused_parameters=True,
        )
        discriminator = DDP(
            discriminator,
            device_ids=[device],
            output_device=rank,
            find_unused_parameters=True,
        )

    # Step 3: training config
    if "recon_loss" in args.autoencoder_train and args.autoencoder_train["recon_loss"] == "l2":
        intensity_loss = MSELoss()
        if rank == 0:
            print("Use l2 loss")
    else:
        intensity_loss = L1Loss()
        if rank == 0:
            print("Use l1 loss")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=args.spatial_dims, network_type="squeeze")
    loss_perceptual.to(device)

    adv_weight = 0.5
    perceptual_weight = args.autoencoder_train["perceptual_weight"]

    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=args.autoencoder_train["lr"] * world_size)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.autoencoder_train["lr"] * world_size)

    # 🐝 Initialize wandb run
    wandb.init(project=f'diff-autoencoder-fov-generation', config=vars(args))
    
    # 🐝 Add training script as an artifact
    artifact_script = wandb.Artifact(name='training', type='file')
    artifact_script.add_file(local_path=os.path.abspath(__file__), name=os.path.basename(__file__))
    wandb.log_artifact(artifact_script)

    # Step 4: training
    autoencoder_warm_up_n_epochs = 5
    n_epochs = args.autoencoder_train["n_epochs"]
    val_interval = args.autoencoder_train["val_interval"]
    best_val_recon_epoch_loss = 100.0
    total_step = 0

    for epoch in range(n_epochs):
        # train
        autoencoder.train()
        discriminator.train()
        if ddp_bool:
            # if ddp, distribute data across n gpus
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        
        generator_loss_list=[]
        loss_d_fake_list=[]
        loss_d_real_list=[]
        recons_loss_list=[]
        p_loss_list=[]
        train_iterator = tqdm(train_loader, desc="Training (G_loss=X.X) (D_loss=X.X)", dynamic_ncols=True)
        for step, batch in enumerate(train_iterator):
            images = batch["image"].to(device)

            # train Generator part
            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, quantization_loss = autoencoder(images)

            recons_loss = intensity_loss(reconstruction, images)
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_g = recons_loss + quantization_loss + perceptual_weight * p_loss

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = loss_g + adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch > autoencoder_warm_up_n_epochs:
                # train Discriminator part
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                optimizer_d.step()

                generator_loss_list.append(generator_loss.mean().item())
                loss_d_fake_list.append(loss_d_fake.mean().item())
                loss_d_real_list.append(loss_d_real.mean().item())

            recons_loss_list.append(recons_loss.mean().item())
            p_loss_list.append(p_loss.mean().item())

            if epoch > autoencoder_warm_up_n_epochs:
                train_iterator.set_description(
                    "Training (G_loss=%2.5f) (D_loss=%2.5f)" % (loss_g.mean().item(), loss_d.mean().item())
                )
            else:
                train_iterator.set_description(
                    "Training (G_loss=%2.5f) (D_loss=X.X)" % (loss_g.mean().item())
                )

        # write train loss for each batch
        # 🐝 Plot G_loss D_loss and discriminator accuracy
        if rank == 0:
            wandb.log({"train_recon_loss/epoch": np.mean(recons_loss_list)})
            wandb.log({"train_perceptual_loss/epoch": np.mean(p_loss_list)})

            if epoch > autoencoder_warm_up_n_epochs:
                wandb.log({"train_adv_loss/epoch": np.mean(generator_loss_list)})
                wandb.log({"train_fake_loss/epoch": np.mean(loss_d_fake_list)})
                wandb.log({"train_real_loss/epoch": np.mean(loss_d_real_list)})

        # validation
        if (epoch) % val_interval == 0:
            autoencoder.eval()
            val_recon_epoch_loss = 0
            for step, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                with torch.no_grad():
                    reconstruction, quantization_loss = autoencoder(images)
                    recons_loss = intensity_loss(
                        reconstruction.float(), images.float()
                    ) + perceptual_weight * loss_perceptual(reconstruction.float(), images.float())

                val_recon_epoch_loss += recons_loss.item()

            val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)
            if rank == 0:
                # save last model
                print(f"Epoch {epoch} val_loss: {val_recon_epoch_loss}")
                if ddp_bool:
                    torch.save(autoencoder.module.state_dict(), trained_g_path_last)
                    torch.save(discriminator.module.state_dict(), trained_d_path_last)
                else:
                    torch.save(autoencoder.state_dict(), trained_g_path_last)
                    torch.save(discriminator.state_dict(), trained_d_path_last)
                # save best model
                if val_recon_epoch_loss < best_val_recon_epoch_loss and rank == 0:
                    best_val_recon_epoch_loss = val_recon_epoch_loss
                    if ddp_bool:
                        torch.save(autoencoder.module.state_dict(), trained_g_path)
                        torch.save(discriminator.module.state_dict(), trained_d_path)
                    else:
                        torch.save(autoencoder.state_dict(), trained_g_path)
                        torch.save(discriminator.state_dict(), trained_d_path)
                    print("Got best val recon loss.")
                    print("Save trained autoencoder to", trained_g_path)
                    print("Save trained discriminator to", trained_d_path)

                # write val loss for each epoch
                # 🐝 Plot G_loss D_loss and discriminator accuracy
                wandb.log({"val_recon_loss/epoch": val_recon_epoch_loss})

                res_img, target_img, pred_img = get_validation_image_diff_2d(images, reconstruction)
                wandb.log({"val_img/last_batch": wandb.Image(res_img, caption=f'res_{epoch}')})
                wandb.log({"val_img/groud_truth": wandb.Image(target_img, caption=f'ground_truth_{epoch}')})
                wandb.log({"val_img/prediction": wandb.Image(pred_img, caption=f'prediction_{epoch}')})


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()