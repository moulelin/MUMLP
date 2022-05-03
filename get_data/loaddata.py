# -*- coding: utf-8 -*-
'''
@File  : loaddata.py
@Author: Moule Lin
@Date  : 2021/6/25 8:53
@Github: https://github.com/moulelin
'''
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import os
import scipy.io as sio
import numpy as np
import torch
trans_image = transforms.Compose([

    transforms.ToTensor(),

])

class getHouston2018(Dataset):
    def __init__(self,dir):
        self.list_houston_2018 = os.listdir(dir)
        self.dir = dir

    def __getitem__(self, item):
        image_name = self.list_houston_2018[item] # get file name
        data_name = os.path.join(self.dir,image_name)
        data = sio.loadmat(data_name)
        train = data["train"]
        label = data["label"]
        data_train = getHouston2018.to_tensor(train)
        data_label = getHouston2018.to_tensor_label(label)
        return data_train, data_label
    def __len__(self):
        return len(self.list_houston_2018)

    @staticmethod
    def to_tensor(data):
        return torch.from_numpy(data).to(torch.float)
    @staticmethod
    def to_tensor_label(data):
        return torch.from_numpy(data).to(torch)
if __name__ == '__main__':
    data_houston = getHouston2018("../Houston_2018_10%/train2")
    dataSet = DataLoader(data_houston,batch_size=5,shuffle=True,num_workers=5,pin_memory=True,drop_last=False)
    print(len(dataSet))
    # for img, label in dataSet:
    trains = iter(dataSet)
    img, label = next(trains)
    print(label)
    #     print(img.shape)
    #     print(label.shape)
