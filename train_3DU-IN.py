# -*- coding: utf-8 -*-
'''
@File  : train_3DU-In.py
@Author: Moule Lin
@Date  : 2021/6/25 8:52
@Github: https://github.com/moulelin
'''
import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from metrics import Indicator
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
import pdb
import numpy as np
import glob
import os
from splitdata_2.spilt import get_data, get_houston2018_dataload


from model_3.se2class import HyperMixerPlus
from model_u.unet_model import UNet


# ------- 1. define loss function --------

criterion_loss = nn.CrossEntropyLoss(reduction='mean')

def muti_cel_loss_fusion(d0,labels_v):

    loss0 = criterion_loss(d0,labels_v)



 #   print("l0: %3f\n"%(loss0.data.item()))

    return loss0


# ------- 2. set the directory of training dataset --------

model_name = 'DU_IN' #'u2netp'

dir = "Houston_2018_10%/train2"


epoch_num = 10000

batch_size = 1000
train, valid, test = get_data.spilt_data()
data_houston_train = get_houston2018_dataload(train)
dataset_train = DataLoader(data_houston_train,batch_size=batch_size,shuffle=True,num_workers=5,pin_memory=True,drop_last=False)

data_houston_valid = get_houston2018_dataload(valid)
dataset_valid = DataLoader(data_houston_valid,batch_size=batch_size,shuffle=True,num_workers=5,pin_memory=True,drop_last=False)

data_houston_test = get_houston2018_dataload(test)
dataset_test = DataLoader(data_houston_test,batch_size=batch_size,shuffle=True,num_workers=5,pin_memory=True,drop_last=False)


# ------- 3. define model --------
# define the net
nb_classes = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = HyperMixerPlus(50, 128, nb_classes) # 输入feature特征，转换之后的特征，最后的类别

model_dir = "save_model/"
def test():

    net.eval()
    running_loss_test = 0.0
    for i, data in enumerate(dataset_test):

        inputs, labels = data
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.LongTensor)
        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.to(device), requires_grad=False), Variable(labels.to(device),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients


        # forward + backward + optimize
        d0 = net(inputs_v)


        loss = muti_cel_loss_fusion(d0, labels_v)


        running_loss_test += loss.data.item()


        # del temporary outputs and loss
        if i and i%10 == 0:
            indicators = Indicator(labels, d0, nb_classes)
            OA = indicators.Over_Accuracy()
            AA = indicators.Average_Accuracy()
            Kappa = indicators.Kappa()
            print("[Test =>  batch: %d, Loss: %3f, OA : %5f, AA : %5f, Kappa : %5f] " % (
                 (i),  running_loss_test / (i + 1), OA, AA, Kappa))
        del d0

def train():

    if torch.cuda.is_available():
        net.to(device=device)

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-8)
    lambdal = lambda epoch: pow((1 - epoch / 10001), 0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdal)
    # ------- 5. training process --------
    print("---start training...")

    net.train()
    for epoch in range(0, epoch_num):
         # 去掉背景数据，这些都是毫无意义的
        running_loss = 0.0
        for i, data in enumerate(dataset_train):



            inputs, labels = data

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.LongTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.to(device), requires_grad=False), Variable(labels.to(device),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0 = net(inputs_v)


            loss = muti_cel_loss_fusion(d0,labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()

         # del temporary outputs and loss


            if i and i%10 == 0:
                indicators = Indicator(labels,d0,nb_classes)
                OA = indicators.Over_Accuracy()
                AA = indicators.Average_Accuracy()
                Kappa = indicators.Kappa()
                print("[Train => epoch: %d/%d, batch: %d/%d, Loss: %3f, OA : %5f, AA : %5f, Kappa : %5f]" % (
                epoch + 1, epoch_num, (i), 400000/batch_size, running_loss / (i+1), OA, AA, Kappa))
            del d0
        scheduler.step()
        if epoch and epoch % 1 == 0:
            torch.save(net.state_dict(), model_dir + model_name+"_itr_%d_train.pth" % (epoch))


        test()
if __name__ == '__main__':
    train()