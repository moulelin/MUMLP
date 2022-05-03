# -*- coding: utf-8 -*-
'''
@File  : test.py
@Author: Moule Lin
@Date  : 2021/7/18 14:50
@Github: https://github.com/moulelin
'''
import torch
import torch.nn as nn
a = torch.randn(1,3,2,2)
linear = nn.Linear(2,5)
print(linear(a).shape)
print(a)
b = range(20,9,-1)
print(list(b))