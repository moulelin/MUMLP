""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
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


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
