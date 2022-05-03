# -*- coding: utf-8 -*-
'''
@File  : test.py
@Author: Moule Lin
@Date  : 2021/7/19 10:37
@Github: https://github.com/moulelin
'''
# import scipy.io as sio
# data = sio.loadmat("Pavia.mat")['pavia']
# data_houston = sio.loadmat("../trans2mat/mat/Houston_2018_train.mat")["Houston_2018"]
# print(data.shape)
# print(data[1,2,:])
# print(data_houston.shape)
# print(data_houston[:,1,2])
import torch
import torch.nn as nn
input = torch.randn(20, 5, 10, 10)
m = nn.LayerNorm(10)
print(m)
