# Intervertebral disc labeling playground

## Description

This repository will be used to test different approach for disc labeling

## See also

NeuroPoly disc labeling implementations:
- Hourglass approach: https://github.com/spinalcordtoolbox/disc-labeling-hourglass
- nnU-Net approach: https://github.com/spinalcordtoolbox/disc-labeling-nnunet
- Disc labeling benchmark: https://github.com/spinalcordtoolbox/disc-labeling-benchmark

## Field of view (FOV) generation using latent diffusion model

> This work is based on this [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) and this [implementation](https://github.com/Project-MONAI/GenerativeModels/tree/main/tutorials/generative/2d_ldm) from MONAI.

### Context

The objective of this project is to reconstruct full MRI scans of the spine from partial FOV using latent diffusion models.

### Training

The training was done using ~60 stiched T2w MRI scans of healthy patients. The models used were a 2d LDM and a 2d VQ-VAE model (used to work in the latent space). To extend the number of images, 5 centered slices were extracted from each T2w scans. 

### Preliminary results

<img width="724" alt="Screenshot 2024-09-11 at 10 35 22" src="https://github.com/user-attachments/assets/5693d6f1-7cc2-420e-bfc7-a197477b9cc5">

In this first result, a cervical scan from a dataset unseen during training was gradually added during the diffusion process (inference) to condition the network to reconstruct the rest of the body. Each of the five images represent a different slice in the 3d input scan (the middle image corresponds to the middle slice of the 3d scan).

<img width="724" alt="Screenshot 2024-09-11 at 10 36 02" src="https://github.com/user-attachments/assets/665014fd-ef65-487c-80c0-27151964cda1">

This other image shows another result when a lumbar scan is used as an input.

### Current limitation

Too few images were available for the training of the LDM. 


