from monai.losses.ssim_loss import SSIMLoss
import torch.nn as nn
import torch
import torch.nn.functional as F

class CriterionCGAN(nn.Module):
    def __init__(self, dim, L1coeff, SSIMcoeff):
        super(CriterionCGAN, self).__init__()
        # Load L1 criterion and parameter
        self.L1criterion = nn.L1Loss()
        self.L1coeff = L1coeff

        # Load SSIM criterion and parameter
        self.SSIMcriterion = SSIMLoss(spatial_dims=dim)
        self.SSIMcoeff = SSIMcoeff
    
    def forward(self, output, target):
        loss = self.L1coeff * self.L1criterion(output, target) + self.SSIMcoeff * self.SSIMcriterion(output, target)
        return loss


def hinge_d_loss(logits_real, logits_fake):
    '''
    Copied from https://github.com/FirasGit/medicaldiffusion/blob/master/vq_gan_3d/model/vqgan.py
    '''
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def gradient_penalty(critic, x, y, y_fake, device="cpu"):
    '''
    Based on https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/4.%20WGAN-GP/utils.py
    '''
    BATCH_SIZE, C, W, H, D = y.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1, 1)).repeat(1, C, W, H, D).to(device)
    interpolated_images = y * alpha + y_fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(x, interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty