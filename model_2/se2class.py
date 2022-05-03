# -*- coding: utf-8 -*-
'''
@File  : se2class.py
@Author: Moule Lin
@Date  : 2021/7/15 18:13
@Github: https://github.com/moulelin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class DimTransBlock(nn.Module): # 这个是用于维度转换，也可以看作是一个mlp，这是第一步
    def __init__(self,in_feature,out_feature):
        super(DimTransBlock, self).__init__()
        self.linear1 = nn.Linear(in_feature, in_feature * 2, bias=True)
        self.linear2 = nn.Linear(in_feature * 2, out_feature, bias=True)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
    def forward(self,x):

        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class AddChannel(nn.Module):  # 这个是用于将只有一个通道的x变为有out_c个通道的数据，这个是第二步
    def __init__(self, out_c,out_feature):
        super(AddChannel, self).__init__()
        self.conv = nn.Conv2d(1,out_c,kernel_size=1) # 采用1*1的卷积，变为out_c个通道，每个通道代表一个信息
        self.norm = nn.LayerNorm(out_feature)
    def forward(self, x):
        x = self.conv(x).squeeze(2)# 将Batch,num_channel,out_feature ，就是叠加通道
        x = self.norm(x)# 将Batch,1,1,out_feature 变为Batch,1,num_channel,out_feature，就是叠加通道
       #  print("+"*10)
       #  print(x.shape)
       #  print("+" * 10)
        return x

class MlpBlock(nn.Module):
    def __init__(self,in_dim,hidden_dim,drop_rate=0):
        super(MlpBlock,self).__init__()
        self.mlp=nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(drop_rate))
    def forward(self,x):

        return self.mlp(x)


class HyperMixBlock(nn.Module):
    '''
       ns:序列数（patch个数）；nc：通道数（嵌入维度）；ds：token mixing的隐层神经元数；dc：channel mixing的隐层神经元数；
      out_feature,num_channel,hidden_dim1,hidden_dimchannel
       '''
    def __init__(self, ns, nc,ds=256, dc=2048, drop_rate=0.):
        super(HyperMixBlock, self).__init__()
        self.norm1 = nn.LayerNorm(ns)
        self.norm2 = nn.LayerNorm(ns)
        self.tokenMix = MlpBlock(in_dim=nc, hidden_dim=dc, drop_rate=drop_rate)
        self.channelMix = MlpBlock(in_dim=ns, hidden_dim=ds, drop_rate=drop_rate)
    def forward(self, x):

        x = self.norm1(x)
        x2 = self.tokenMix(x.transpose(1, 2)).transpose(1, 2)  # 不能用.T,否则b会跑到最后一维
        x = x + x2
        x2 = self.norm2(x)
        x2 = self.channelMix(x2)
        return x + x2

class HyperMixerPlus(nn.Module):
    def __init__(self,in_feature,out_feature,num_classes,num_channel = 64,num_layers = 16,hidden_dim1 = 526,hidden_dimchannel = 128):
        super(HyperMixerPlus, self).__init__()
        self.dimtrans = DimTransBlock(in_feature,out_feature) # 增加特征长度，防止有些数据的channel特别小
        self.addC = AddChannel(num_channel,out_feature) # 和Mixer一样，堆叠channel，这里的channel是由原始的一个生成的，变成batch*64*len
        hypermixblock = HyperMixBlock(out_feature,num_channel,hidden_dim1,hidden_dimchannel) # out_feature就是序列个数
        self.mixlayers = nn.Sequential(*[hypermixblock for _ in range(num_layers)])
        self.norm = nn.LayerNorm(out_feature)
        self.cls = nn.Linear(out_feature, num_classes)
    def forward(self,x):
        x = self.dimtrans(x).unsqueeze(1).unsqueeze(1) # 输入为Batch,in_feature=>Batch,1,1,out_feature
        x = self.addC(x) # Batch,1,1,out_feature=>Batch,num_channel,out_feature
        x = self.mixlayers(x)
        x = self.norm(x)
        x = torch.mean(x, dim=1)  # 逐通道求均值 N C
        x = self.cls(x)
        return x












class InVol(nn.Module):
    def __init__(self, in_c, out_c, r=2, k = 3, ss = 1, G = 8): #c/G个通道共享一个参数，这里是内卷积操作，后期应该修改一部分
        super(InVol, self).__init__()
        self.o = nn.AvgPool2d(ss, ss) if ss > 1 else nn.Identity()
        self.conv_reduce = nn.Conv2d(in_c, in_c // r, 1)
        self.conv_span = nn.Conv2d(in_c // r, k * k * G, 1)
        self.unfold = nn.Unfold(k, dilation=1, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        y = x
        b, c, h, w = x.shape
        x_unfolded = self.unfold(x)  # B,CxKxK,HxW
        x_unfolded = x_unfolded.view(b, self.G, c // self.G, self.k * self.k, h, w) # k*k,h,w 就是一个通道形成的
        # 这里可以把k*k,h,w看为之前数据一个通道形成的，一共有c个通道，每个通道都会形成k*k,h,w，
        # 这里是把卷积操作（3*3）进行了拉直，而不是和论文中的把通道拉直，结果是一样的，这样会方便很多
        kernel = self.conv_span(self.conv_reduce(self.o(x)))# 把图像维度变为k * k * G
        kernel = kernel.view(b, self.G, self.k * self.k, h, w).unsqueeze(2)#从通道维度进行分割
        # 这里相当于把k*k个通道和一个通道形成的unfolded相乘，就是相当于每个点的通道9 和周围3*3领域点相乘
        # kernel是通道的9个点，x_unfolded是领域的9个点
        # 这里需要广播，就是把一个通道变为c/G个，就是一个点的通道需要和他对应的每个通道形成的k*k,h,w的领域相乘（这里不是这样子的）。最后累加每个通道的结果
        # 这里如果kernel也是c个通道，就是上门说的那样，但是这里通道变为了G*k*k,所以就是说一个点的通道和c/G个它生成的unfolded相乘就行
        # 假设c = 32, G*k*k=16，G=2. 就是x每个点对应通道和16个相乘再相加就行，原来应该是每个点通道和形成的，现在就一半，减少计算量
        out = torch.mul(kernel, x_unfolded).sum(dim=3)
        out = out.view(b, c, h, w)
        return out




