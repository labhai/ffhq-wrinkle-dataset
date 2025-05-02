"""
Parts (building blocks) of the U-Net model:
    - DoubleConv
    - Down (maxpool + conv)
    - Up (upsample + conv)
    - OutConv (final 1x1 convolution)

No specific license indicated here, presumably open-source base.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (Conv -> BN -> ReLU) * 2

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        mid_channels (int): optional intermediate channels for first conv layer.

    Returns:
        torch.Tensor: features after double convolution.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv.

    Args:
        in_channels (int): input channel dimension.
        out_channels (int): output channel dimension.

    Returns:
        torch.Tensor: downsampled feature.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv.

    Args:
        in_channels (int): channels of the input feature.
        out_channels (int): desired output feature channels after upsampling.
        bilinear (bool): if True, use bilinear upsampling, else transposed conv.

    Returns:
        torch.Tensor: upsampled and merged feature.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal conv to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # pad x1 to match x2
        x1 = F.pad(x1,
                   [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 convolution layer to get the required number of output channels.

    Args:
        in_channels (int): input channel dimension.
        out_channels (int): output channel dimension.
        kernel_size (int): size of the convolution kernel, defaults to 1.
        padding (int): padding for the convolution, defaults to 0.

    Returns:
        torch.Tensor: final segmentation logits.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super(OutConv, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)
