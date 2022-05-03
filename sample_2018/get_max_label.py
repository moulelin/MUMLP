# -*- coding: utf-8 -*-
'''
@File  : get_max_label.py
@Author: Moule Lin
@Date  : 2021/6/25 15:30
@Github: https://github.com/moulelin
'''
import scipy.io as sio
import numpy as np
ground_truth_compl = sio.loadmat("../trans2mat/mat/Houston_2018_label.mat")["Houston_2018"]
print(len(ground_truth_compl.flatten()))
print(np.sum(np.where(ground_truth_compl == 0,1,0)))