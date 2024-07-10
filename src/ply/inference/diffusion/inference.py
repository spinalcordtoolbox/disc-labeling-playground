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
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast
from generative.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import set_determinism, first
from PIL import Image
from ply.train.diffusion.utils import prepare_dataloader_inference, prepare_dataloader, define_instance
from visualize_image import visualize_2d_image

from ply.models.diffusion.ldm import LatentDiffusionInferer
from ply.models.diffusion.vqvae import VQVAE
from ply.utils.plot import get_validation_image_diff_2d


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_32g.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument(
        "-img",
        type=str,
        required=True,
        help="Image with a partial fov",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    set_determinism(42)

    # Input image path
    input_path = args.img

    # Load image
    inf_loader = prepare_dataloader_inference(
        img_path=input_path,
        v_patch_size=args.diffusion_train["val_patch_size"],
        amp=True,
        sample_axis=args.sample_axis,
        cache=0.0,
        num_center_slice=5,
    )

    # load trained networks
    # Load VQVAE
    autoencoder = VQVAE(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=args.autoencoder_def["num_channels"],
        num_res_channels=args.autoencoder_def["num_channels"][-1],
        num_res_layers=args.autoencoder_def["num_res_blocks"],
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim
    ).to(device)

    trained_g_path = os.path.join(args.model_dir, "vqvae.pt")
    autoencoder.load_state_dict(torch.load(trained_g_path))

    # Compute Scaling factor
    size_divisible = 2 ** (len(args.autoencoder_def["num_channels"]) + len(args.diffusion_def["num_channels"]) - 2)
    train_loader, _ = prepare_dataloader(
        args,
        args.diffusion_train["batch_size"],
        args.diffusion_train["train_patch_size"],
        args.diffusion_train["val_patch_size"],
        sample_axis=args.sample_axis,
        randcrop=True,
        cache=0.0,
        download=False,
        size_divisible=size_divisible,
        amp=True,
        train_transform='full',
        val_transform='full',
        inf=True
    )

    with torch.no_grad():
        with autocast(enabled=True):
            check_data = first(train_loader)
            z = autoencoder.encode_stage_2_inputs(check_data["image"].to(device), quantize=False)
            print(f"Latent feature shape {z.shape}")
            print(f"Scaling factor set to {1/torch.std(z)}")
    scale_factor = 1 / torch.std(z)

    # Load diffusion model
    diffusion_model = define_instance(args, "diffusion_def").to(device)
    trained_diffusion_path = os.path.join(args.model_dir, "diffusion_unet_last.pt")
    diffusion_model.load_state_dict(torch.load(trained_diffusion_path))

    scheduler = DDPMScheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=args.NoiseScheduler["beta_start"],
        beta_end=args.NoiseScheduler["beta_end"],
    )
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    for step, batch in enumerate(inf_loader):
        images, masks = batch["image"].to(device), batch["mask"].to(device)

        # Generate random noise
        noise_shape = [images.shape[0]] + list(z.shape[1:])
        noise = torch.randn(noise_shape, dtype=images.dtype).to(device)

        with torch.no_grad():
            with autocast(enabled=True):
                synthetic_images = inferer.paint(
                    inputs=images,
                    noise=noise,
                    mask=masks,
                    autoencoder_model=autoencoder,
                    diffusion_model=diffusion_model,
                    scheduler=scheduler,
                )
        
        filename = os.path.join(args.output_dir, os.path.basename(input_path).replace(".nii.gz", "") + "_extended" + ".jpeg")
        res_img, target_img, pred_img = get_validation_image_diff_2d(images, synthetic_images, mask=masks.astype(bool))
        img = Image.fromarray(visualize_2d_image(res_img), "RGB")

        # Create output directory
        out_dir = os.path.dirname(filename)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Save output
        img.save(filename)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()