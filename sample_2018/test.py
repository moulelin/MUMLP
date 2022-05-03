# -*- coding: utf-8 -*-
'''
@File  : test.py
@Author: Moule Lin
@Date  : 2021/6/23 19:45
@Github: https://github.com/moulelin
'''

import scipy.io as sio
import numpy as np
#E:\python\3DU-IN\trans2mat\mat\Houston_2018_train.mat
data = sio.loadmat("../trans2mat/mat/Houston_2018_train.mat")["Houston_2018"]
ground_truth = sio.loadmat("../trans2mat/mat/Houston_2018_label.mat")["Houston_2018"].reshape(1,-1)
ground_truth = np.squeeze(ground_truth, 0)

print(min(ground_truth))
