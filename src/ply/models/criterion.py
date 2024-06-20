from monai.losses.ssim_loss import SSIMLoss
import torch.nn as nn
import torch

class CriterionCGAN(nn.Module):
    def __init__(self, dim, L1coeff, SSIMcoeff, KLcoeff=0):
        super(CriterionCGAN, self).__init__()
        # Load L1 criterion and parameter
        self.L1criterion = nn.L1Loss()
        self.L1coeff = L1coeff

        # Load SSIM criterion and parameter
        self.SSIMcriterion = SSIMLoss(spatial_dims=dim)
        self.SSIMcoeff = SSIMcoeff

        # Load KL loss
        self.KLcoeff = KLcoeff
    
    def forward(self, output, target, mu=0, sigma=0):
        if self.KLcoeff != 0:
            loss = self.L1coeff * self.L1criterion(output, target) + self.SSIMcoeff * self.SSIMcriterion(output, target) + self.KLcoeff * KL_loss(mu, sigma)
        else:
            loss = self.L1coeff * self.L1criterion(output, target) + self.SSIMcoeff * self.SSIMcriterion(output, target)
        return loss

##
def KL_loss(z_mu, z_sigma):
    '''
    Based on https://github.com/Project-MONAI/tutorials/blob/main/generative/2d_ldm/utils.py
    '''
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]