""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
#
class INVOL(nn.Module):
    def __init__(self,c_in,k=3,r = 2,ss=1):
        super(INVOL,self).__init__()
        self.G = 8
        self.k = k
        self.r = r
        self.o = nn.AvgPool2d(ss, ss) if ss > 1 else nn.Identity()
        self.conv_reduce = nn.Conv2d(c_in,c_in//r,1)
        self.conv_span = nn.Conv2d(c_in//r,k*k*self.G,1)
        self.unfold = nn.Unfold(k, dilation=1, padding=1, stride=1)
    def forward(self,x):
        b,c,h,w = x.shape
        x_unfolded = self.unfold(x) # B,CxKxK,HxW
        x_unfolded = x_unfolded.view(b,self.G,c//self.G,self.k*self.k,h,w)
        kernel = self.conv_span(self.conv_reduce(self.o(x)))
        kernel = kernel.view(b, self.G, self.k * self.k, h, w).unsqueeze(2)
        out = torch.mul(kernel, x_unfolded).sum(dim=3)
        out = out.view(b,c,h,w)
        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
          #  INVOL(mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            INVOL(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
