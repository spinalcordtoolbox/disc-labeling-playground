"""
Based on https://github.com/Project-MONAI/MONAI/blob/46a5272196a6c2590ca2589029eed8e4d56ff008/monai/networks/nets/attentionunet.py#L185-L290 
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
from functools import reduce

from monai.networks.blocks.convolutions import Convolution

from monai.networks.nets.attentionunet import ConvBlock, AttentionLayer
from monai.networks.nets.fullyconnectednet import FullyConnectedNet

class AttentionUnet(nn.Module):
    """    
    Attention Unet based on
    Otkay et al. "Attention U-Net: Learning Where to Look for the Pancreas"
    https://arxiv.org/abs/1804.03999

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_shape: input image shape for MLP.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        channels (Sequence[int]): sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides (Sequence[int]): stride to use for convolutions.
        kernel_size: convolution kernel size.
        up_kernel_size: convolution kernel size for transposed convolution layers.
        dropout: dropout ratio. Defaults to no dropout.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_shape: Sequence[int],
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dimensions = spatial_dims
        self.in_shape = in_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.dropout = dropout

        head = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=channels[0],
            dropout=dropout,
            kernel_size=self.kernel_size,
        )
        reduce_channels = Convolution(
            spatial_dims=spatial_dims,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
            conv_only=True,
        )
        self.up_kernel_size = up_kernel_size

        # MLP parameters
        self.flatten_channels = int((reduce(lambda x, y: x * y, self.in_shape)/2**(3*(len(self.channels)-1)))*self.channels[-1])
        self.hidden_channels = [self.flatten_channels*2, self.flatten_channels*2]

        def _create_block(channels: Sequence[int], strides: Sequence[int]) -> nn.Module:
            if len(channels) > 2:
                subblock = _create_block(channels[1:], strides[1:])
                return AttentionLayer(
                    spatial_dims=spatial_dims,
                    in_channels=channels[0],
                    out_channels=channels[1],
                    submodule=nn.Sequential(
                        ConvBlock(
                            spatial_dims=spatial_dims,
                            in_channels=channels[0],
                            out_channels=channels[1],
                            strides=strides[0],
                            dropout=self.dropout,
                            kernel_size=self.kernel_size,
                        ),
                        subblock,
                    ),
                    up_kernel_size=self.up_kernel_size,
                    strides=strides[0],
                    dropout=dropout,
                )
            else:
                # the next layer is the bottom so stop recursion,
                # create the bottom layer as the subblock for this layer
                return self._get_bottom_layer(channels[0], channels[1], strides[0])

        encdec = _create_block(self.channels, self.strides)
        self.model = nn.Sequential(head, encdec, reduce_channels)

    def _get_bottom_layer(self, in_channels: int, out_channels: int, strides: int) -> nn.Module:
        return AttentionLayer(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            submodule=nn.Sequential(
                ConvBlock(
                    spatial_dims=self.dimensions,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    dropout=self.dropout,
                    kernel_size=self.kernel_size,
                ),
                MLPBlock(
                    in_channels=self.flatten_channels,
                    out_channels=self.flatten_channels,
                    hidden_channels=self.hidden_channels,
                    dropout=self.dropout,
                ),
            ),
            up_kernel_size=self.up_kernel_size,
            strides=strides,
            dropout=self.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_m: torch.Tensor = self.model(x)
        return x_m


class MLPBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Sequence[int],
        dropout=0.0,
    ):
        super().__init__()
        
        self.mlp = FullyConnectedNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input
        in_shape = x.shape
        x_flat = torch.flatten(x, start_dim=1)

        # Pass through MLP layers
        x_flat = self.mlp(x_flat)

        # Convert back to original shape
        x_out = torch.reshape(x_flat, in_shape)
        return x_out
