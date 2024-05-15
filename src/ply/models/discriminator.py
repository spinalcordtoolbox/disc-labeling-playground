'''
Based on https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/Pix2Pix/discriminator_model.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[4,4,4], stride=[2,2,2], padding=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

                      
class Discriminator(nn.Module):
    def __init__ (self, in_channels=1, features=[16, 32, 64, 128], kernel_size=[4,4,4]):
        super(Discriminator, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv3d(
                in_channels * 2,
                features[0],
                kernel_size=kernel_size,
                stride=[2,2,2],
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for i, feature in enumerate(features[1:-1]):
            layers.append(
                CNNBlock(in_channels, feature, kernel_size=kernel_size, stride=[2,2,2], padding=2 if feature == features[i+1] else 1)
                )
            in_channels = feature
        layers.append(
                CNNBlock(in_channels, features[-1], kernel_size=kernel_size, stride=[1,1,1])
                )
        in_channels = feature

        layers.append(
            nn.Conv3d(in_channels, 1, kernel_size=kernel_size, stride=[1,1,1], padding=1, padding_mode="reflect")
            )

        self.model = nn.Sequential(*layers)
    
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        
        return x.squeeze()