# -*- coding: utf-8 -*-
'''
@File  : sample_bfs_group_test.py
@Author: Moule Lin
@Date  : 2021/6/23 19:43
@Github: https://github.com/moulelin
'''

import scipy.io as sio

from collections import Counter
import numpy as np
from random import randint
import pickle
import queue
import pandas as pd
# next_step = [[0, 1],  # 向右走
#              [1, 0],  # 向下走
#              [0, -1],  # 向左走
#              [-1, 0],  # 向上走
#              [-1, -1], # 左下角
#              [1,1],# 右上角
#              [-1,1],# 左上角
#              [1,-1],# 右下角
#              ]
next_step=[
[-1,1],# 左上角
[-1, 0],  # 向上走
[1,1],# 右上角
[0, -1],  # 向左走
[0, 1],  # 向右走
[-1, -1], # 左下角
[1, 0],  # 向下走
[1,-1],# 右下角
]
w_sample_ = 300
h_sample_ = 300
def sampling_group_class(ground_truth,data,ignore_class=None):
    """

    :param ground_truth: image label
    :param data:
    :param ignore_class: e.g. 0
    :return:
    group each class and divide into
    """

    c,w,h = data.shape
    w_random = randint(14,w-14)
    h_random = randint(14,h-14)
    point_FIFO = queue.Queue() # 新建一个队列，类似广度优先搜索
    w_sample = 200
    h_sample = 200

    visit = np.zeros((w+1,h+1), dtype=np.int) # 标记数组，标记是否已经入队列了，
    point_FIFO.put((w_random,h_random)) # 先加入中心点
    visit[w_random][h_random] = 1 # visit数组标记

    while(True): # 重新找到随机点中心
        temp_index = point_FIFO.get()
        if ground_truth[temp_index[0]][temp_index[1]] == 0.: # 重新找到新的中心点
            for i in next_step: # 八个方向
                if visit[temp_index[0]+i[0]][temp_index[1]+i[1]] == 0: # 防止bfs出边界，导致截图时超出边界
                    if temp_index[0]+i[0]>14 and temp_index[0]+i[0]<588 and temp_index[1]+i[1]<370 and temp_index[1]+i[1]>14:
                        point_FIFO.put((temp_index[0]+i[0],temp_index[1]+i[1]))
                        visit[temp_index[0]+i[0]][temp_index[1]+i[1]] = 1
                        #print(i)
        else:

            w = temp_index[0]
            h = temp_index[1]+2000

            break
    return w,h
def pre_process_data(data, norm_dim):
    """
    :param data: np.array, original  data without normalization.
    :param norm_dim: int, normalization dimension. we use 2 dim(channel) to normalize
    :return:
        norm_base: list, [max_data, min_data], data of normalization base.
        norm_data: np.array, normalized traffic data.
    """
    norm_base = normalize_base(data, norm_dim)  # find the normalize base
    norm_data = normalize_data(norm_base[0], norm_base[1], data)  # normalize data

    return norm_base, norm_data


def normalize_base(data, norm_dim):
    """
    :param data: np.array, original  data without normalization.
    :param norm_dim: int, normalization dimension.
    :return:
        max_data: np.array
        min_data: np.array
    """

    max_data = np.max(data, norm_dim, keepdims=True)  # [N, T, D] , norm_dim=1, [N, 1, D]
    min_data = np.min(data, norm_dim, keepdims=True)

    return max_data, min_data

def normalize_data(max_data, min_data, data):
    """
    :param max_data: np.array, max data.
    :param min_data: np.array, min data.
    :param data: np.array, original  data without normalization.
    :return:
        np.array, normalized traffic data.
    """
    mid = min_data
    base = max_data - min_data
    normalized_data = (data - mid) / (base+0.01)

    return normalized_data

def sixth_point_build_dataset():

    data = sio.loadmat("../trans2mat/mat/Houston_2018_train.mat")["Houston_2018"][:,:601,2000:]
    ground_truth = sio.loadmat("../trans2mat/mat/Houston_2018_label.mat")["Houston_2018"][:601,2000:]

    data_compl = sio.loadmat("../trans2mat/mat/Houston_2018_train.mat")["Houston_2018"]
    ground_truth_compl = sio.loadmat("../trans2mat/mat/Houston_2018_label.mat")["Houston_2018"]

    print(data.shape)
    c,w,h= data.shape
    sample_mat = np.zeros((c,w_sample_,w_sample_),dtype=np.float32)
    sample_gt = np.zeros((w_sample_,w_sample_),dtype=np.int)
    for epoch in range(1,51):
        for i in range(12): #64group
            for j in range(12):
                w_,h_ = sampling_group_class(ground_truth,data)
              #  print(w_, h_)
                temp_data = ground_truth_compl[w_-12:w_+13,h_-12:h_+13]
                statistics = len(temp_data[temp_data!=0.])
               # print(statistics)
                if statistics>=60:
                    # print(w_,h_)
                    sample_mat[:,i*25:(i+1)*25,j*25:(j+1)*25] = data_compl[:,w_-12:w_+13,h_-12:h_+13]
                    sample_gt[i * 25:(i + 1) * 25, j * 25:(j + 1) * 25] = ground_truth_compl[w_ - 12:w_ + 13,
                                                                              h_ - 12:h_ + 13]
                else:
                    while(True):
                        w_2,h_2 = sampling_group_class(ground_truth, data)
                        temp_data = ground_truth_compl[w_2 - 12:w_2 + 13, h_2 - 12:h_2 + 13]
                       # print(temp_data)
                        statistics = len(temp_data[temp_data != 0.])
                        #    print(statistics)
                        # print(w_2,h_2)
                        if statistics >= 60:
                            sample_mat[:,i * 25:(i + 1) * 25, j * 25:(j + 1) * 25] = data_compl[:,w_2 - 12:w_2 + 13,h_2 - 12:h_2 + 13]
                            sample_gt[i * 25:(i + 1) * 25, j * 25:(j + 1) * 25] = ground_truth_compl[w_2 - 12:w_2 + 13,h_2 - 12:h_2 + 13]
                            break
        _, sample_mat = pre_process_data(sample_mat, -1)
    # temp = pd.DataFrame(sample_mat)
    # temp.to_pickle("Houston_10%/sample_houston_10.pt2", compression="xz")

        sio.savemat(f"../Houston_2018_10%/test/sample_houston_test_{str(epoch)}.mat",{"train":sample_mat,"label":sample_gt},do_compression = True)
        print(epoch)
    # with open("Houston_10%/sample_houston_10.pt",'wb') as f:
    #     pickle.dump({"train":sample_mat,"label":sample_gt},f,protocol=pickle.HIGHEST_PROTOCOL)
    #
    # print(sample_mat)
    # print(sample_mat.shape)
    # print(sample_gt.shape)



if __name__ == '__main__':
    sixth_point_build_dataset()


