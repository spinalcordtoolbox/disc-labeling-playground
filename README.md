# Intervertebral disc labeling playground

## Description

This repository will be used to test different approach for disc labeling

## See also

NeuroPoly disc labeling implementations:
- Hourglass approach: https://github.com/spinalcordtoolbox/disc-labeling-hourglass
- nnU-Net approach: https://github.com/spinalcordtoolbox/disc-labeling-nnunet
- Disc labeling benchmark: https://github.com/spinalcordtoolbox/disc-labeling-benchmark

## Field of view (FOV) generation using latent diffusion model

### Context

The objective of this project is to reconstruct full MRI scans of the spine from partial FOV using latent diffusion models.

### Training

The training was done using ~60 stiched T2w MRI scans of healthy patients. The models used were a 2D LDM and a VQ-VAE model. To extend the number of images, 5 centered slices were extracted from each T2w scans. 

### Preliminary results

<img width="1068" alt="Screenshot 2024-09-11 at 10 35 22" src="https://github.com/user-attachments/assets/5693d6f1-7cc2-420e-bfc7-a197477b9cc5">
<img width="1069" alt="Screenshot 2024-09-11 at 10 36 02" src="https://github.com/user-attachments/assets/665014fd-ef65-487c-80c0-27151964cda1">
