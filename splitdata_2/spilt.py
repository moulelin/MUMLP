# -*- coding: utf-8 -*-
'''
@File  : spilt.py
@Author: Moule Lin
@Date  : 2021/7/19 9:20
@Github: https://github.com/moulelin
'''
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import os
import torch
train_num = 400000 # 总共504712
valid_num = 50000

class get_data():
    @staticmethod
    def spilt_data():
        ground_truth = sio.loadmat("trans2mat/mat/Houston_2018_label.mat")["Houston_2018"]
        sample_index = []
        for r in range(ground_truth.shape[0]):
            for c in range(ground_truth.shape[1]):
                if ground_truth[r, c] > 0:
                   sample_index.append([r, c, ground_truth[r, c]-1]) # 减1是为了分为0-19而不是1-20

        sample_index = np.array(sample_index)
        index = np.arange(sample_index.shape[0])
        np.random.shuffle(index)
        sample_index_train = sample_index[index[:train_num]] # 400000
        sample_index_valid = sample_index[index[train_num:train_num+valid_num]] #400000-450000
        sample_index_test = sample_index[index[train_num+valid_num:]]#450000-504712
        print(sample_index.shape[0])
        label_num = len(np.unique(sample_index[:, 2]))
        print(np.unique(sample_index[:, 2]))

        return sample_index_train, sample_index_valid, sample_index_test
    """
    (504712,)
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    """




trans_image = transforms.Compose([

    transforms.ToTensor(),

])

class get_houston2018_dataload(Dataset):
    def __init__(self, data):
        self.data = data
        houston_data = sio.loadmat("trans2mat/mat/Houston_2018_train.mat")["Houston_2018"]
        _,self.houston_data = get_houston2018_dataload.pre_process_data(houston_data,norm_dim=0)

    def __getitem__(self, item):

        index = self.data[item]
        train = self.houston_data[:,index[0],index[1]]

        label = np.array(index[2])

        data_train = get_houston2018_dataload.to_tensor(train)
        data_label = get_houston2018_dataload.to_tensor_label(label)
        return data_train, data_label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def to_tensor(data):
        return torch.from_numpy(data).to(torch.float)
    @staticmethod
    def to_tensor_label(data):
        return torch.from_numpy(data).to(torch.int32)

    @staticmethod
    def pre_process_data(data, norm_dim):
        """
        :param data: np.array, original  data without normalization.
        :param norm_dim: int, normalization dimension. we use 2 dim(channel) to normalize
        :return:
            norm_base: list, [max_data, min_data], data of normalization base.
            norm_data: np.array, normalized traffic data.
        """
        norm_base = get_houston2018_dataload.normalize_base(data, norm_dim)  # find the normalize base
        norm_data = get_houston2018_dataload.normalize_data(norm_base[0], norm_base[1], data)  # normalize data

        return norm_base, norm_data

    @staticmethod
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

    @staticmethod
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
        normalized_data = (data - mid) / (base + 0.01)

        return normalized_data

if __name__ == '__main__':
    train,valid,test = get_data.spilt_data()
    data_houston = get_houston2018_dataload(train)
    dataSet = DataLoader(data_houston, batch_size=1, shuffle=True, num_workers=5, pin_memory=True, drop_last=False)
    print(len(dataSet))
    for train_,label_ in dataSet:
        print("+++")
        print(train_)
        print(label_)
