'''
Based on https://github.com/cihanongun/3D-CGAN/
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

                      
class Discriminator(nn.Module):
    def __init__ (self, in_channels=1, feature_size=32, kernel_size=[4,4,4]):
        super(Discriminator, self).__init__()
                
        self.disc_conv1 = nn.Conv3d(in_channels, feature_size, kernel_size=kernel_size, stride=[2,2,2], padding=1)
        self.disc_conv2 = nn.Conv3d(feature_size, feature_size*2, kernel_size=kernel_size, stride=[2,2,2], padding=1)
        self.disc_conv3 = nn.Conv3d(feature_size*2, feature_size*4, kernel_size=kernel_size, stride=[2,2,2], padding=1)
        self.disc_conv4 = nn.Conv3d(feature_size*4, feature_size*8, kernel_size=kernel_size, stride=[2,2,2], padding=1)
        self.disc_conv5 = nn.Conv3d(feature_size*8, 1, kernel_size=kernel_size, stride=[2,2,2], padding=1)
        
        self.disc_bn1 = nn.BatchNorm3d(feature_size)
        self.disc_bn2 = nn.BatchNorm3d(feature_size*2)
        self.disc_bn3 = nn.BatchNorm3d(feature_size*4)
        self.disc_bn4 = nn.BatchNorm3d(feature_size*8)
        
        self.LRelu = nn.LeakyReLU(0.2, True)
    
    def forward(self, x, y):
        
        x = torch.cat([x, y], dim=1)
        
        x = self.LRelu(self.disc_bn1(self.disc_conv1(x)))
        x = self.LRelu(self.disc_bn2(self.disc_conv2(x)))
        x = self.LRelu(self.disc_bn3(self.disc_conv3(x)))
        x = self.LRelu(self.disc_bn4(self.disc_conv4(x)))
        x = self.disc_conv5(x)
        x = torch.sigmoid(x)
        
        return x.squeeze()