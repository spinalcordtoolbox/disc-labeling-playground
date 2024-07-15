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

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import Inferer
from monai.utils import optional_import
from generative.inferers import DiffusionInferer

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

class LatentDiffusionInferer(DiffusionInferer):
    """
    LatentDiffusionInferer takes a stage 1 model (VQVAE or AutoencoderKL), diffusion model, and a scheduler, and can
    be used to perform a signal forward pass for a training iteration, and sample from the model.

    Args:
        scheduler: a scheduler to be used in combination with `unet` to denoise the encoded image latents.
        scale_factor: scale factor to multiply the values of the latent representation before processing it by the
            second stage.
    """

    def __init__(self, scheduler: nn.Module, scale_factor: float = 1.0) -> None:
        super().__init__(scheduler=scheduler)
        self.scale_factor = scale_factor

    def __call__(
        self,
        inputs: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: input image to which the latent representation will be extracted and noise is added.
            autoencoder_model: first stage model.
            diffusion_model: diffusion model.
            noise: random noise, of the same shape as the latent representation.
            timesteps: random timesteps.
            condition: conditioning for network input.
        """
        with torch.no_grad():
            latent = autoencoder_model.encode_stage_2_inputs(inputs, quantize=False) * self.scale_factor

        prediction = super().__call__(
            inputs=latent, diffusion_model=diffusion_model, noise=noise, timesteps=timesteps, condition=condition
        )

        return prediction

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        verbose: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired latent representation.
            autoencoder_model: first stage model.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            verbose: if true, prints the progression bar of the sampling process.
        """
        outputs = super().sample(
            input_noise=input_noise,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            intermediate_steps=intermediate_steps,
            conditioning=conditioning,
            verbose=verbose,
        )

        if save_intermediates:
            latent, latent_intermediates = outputs
        else:
            latent = outputs

        image = autoencoder_model.decode_stage_2_outputs(latent / self.scale_factor)

        if save_intermediates:
            intermediates = []
            for latent_intermediate in latent_intermediates:
                intermediates.append(autoencoder_model.decode_stage_2_outputs(latent_intermediate / self.scale_factor))
            return image, intermediates

        else:
            return image

    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        conditioning: torch.Tensor | None = None,
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
        verbose: bool = True,
        resample_latent_likelihoods: bool = False,
        resample_interpolation_mode: str = "nearest",
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Computes the log-likelihoods of the latent representations of the input.

        Args:
            inputs: input images, NxCxHxW[xD]
            autoencoder_model: first stage model.
            diffusion_model: model to compute likelihood from
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
            resample_latent_likelihoods: if true, resamples the intermediate likelihood maps to have the same spatial
                dimension as the input images.
            resample_interpolation_mode: if use resample_latent_likelihoods, select interpolation 'nearest', 'bilinear',
                or 'trilinear;
        """
        if resample_latent_likelihoods and resample_interpolation_mode not in ("nearest", "bilinear", "trilinear"):
            raise ValueError(
                f"resample_interpolation mode should be either nearest, bilinear, or trilinear, got {resample_interpolation_mode}"
            )
        latents = autoencoder_model.encode_stage_2_inputs(inputs, quantize=False) * self.scale_factor
        outputs = super().get_likelihood(
            inputs=latents,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            conditioning=conditioning,
            verbose=verbose,
        )
        if save_intermediates and resample_latent_likelihoods:
            intermediates = outputs[1]
            resizer = nn.Upsample(size=inputs.shape[2:], mode=resample_interpolation_mode)
            intermediates = [resizer(x) for x in intermediates]
            outputs = (outputs[0], intermediates)
        return outputs
    
    @torch.no_grad()
    def paint(
        self,
        inputs: torch.Tensor,
        noise: torch.Tensor,
        mask: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        verbose: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Generate a sample from a partially noised input

        Args:
            input: input image that will be partially noised.
            noise: random noise, of the same shape as the latent representation.
            mask: mask of the region which should not change
            autoencoder_model: first stage model.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            verbose: if true, prints the progression bar of the sampling process.
        """
        with torch.no_grad():
            latent = autoencoder_model.encode_stage_2_inputs(inputs, quantize=False) * self.scale_factor

        # Downsample mask to latent shape
        shape = latent.shape
        latent_mask = F.interpolate(mask, (shape[2], shape[3]))
        latent_mask = latent_mask.repeat(1, shape[1], 1, 1) # Repeat mask along the channels dimension

        # Initialise out_image with input noise
        out_image = scheduler.add_noise(original_samples=latent, noise=noise, timesteps=scheduler.timesteps[0])

        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        for t in progress_bar:
            # Generate noisy input (forward process)
            noisy_image = scheduler.add_noise(original_samples=latent, noise=noise, timesteps=(t))

            # 1. predict noise model_output
            model_output = diffusion_model(
                out_image, timesteps=torch.Tensor((t,)).to(out_image.device), context=conditioning
            )
            # 2. compute previous image: x_t -> x_t-1
            # Based on https://dl.acm.org/doi/abs/10.1145/3592450
            diff_image, _ = scheduler.step(model_output, t, out_image)
            out_image = diff_image * (1 - latent_mask) + noisy_image * latent_mask

            # Save intermediate steps
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(out_image)

        if save_intermediates:
            latent = out_image
            latent_intermediates = intermediates
        else:
            latent = out_image

        image = autoencoder_model.decode_stage_2_outputs(latent / self.scale_factor)

        if save_intermediates:
            intermediates = []
            for latent_intermediate in latent_intermediates:
                intermediates.append(autoencoder_model.decode_stage_2_outputs(latent_intermediate / self.scale_factor))
            return image, intermediates

        else:
            return image