from monai.losses.ssim_loss import SSIMLoss
import torch.nn as nn

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