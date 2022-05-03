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

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
import pdb
import numpy as np
import glob
import os
from get_data.loaddata import getHouston2018


from model.D3U_IN import DU_IN


# ------- 1. define loss function --------

criterion_loss = nn.CrossEntropyLoss(size_average=True)

def muti_cel_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    loss0 = criterion_loss(d0,labels_v)
    loss1 = criterion_loss(d1,labels_v)
    loss2 = criterion_loss(d2,labels_v)
    loss3 = criterion_loss(d3,labels_v)
    loss4 = criterion_loss(d4,labels_v)
    loss5 = criterion_loss(d5,labels_v)
    loss6 = criterion_loss(d6,labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

    return loss0, loss


# ------- 2. set the directory of training dataset --------

model_name = 'DU_IN' #'u2netp'

dir = "Houston_2018_10%/train2"


epoch_num = 10000
batch_size_train = 20
batch_size_val = 1
train_num = 0
val_num = 0
model_dir = "save_model/"


data_houston = getHouston2018("Houston_2018_10%/train2")
dataSet = DataLoader(data_houston,batch_size=20,shuffle=True,num_workers=5,pin_memory=True,drop_last=False)

data_houston_test = getHouston2018("Houston_2018_10%/test")
dataSetTest = DataLoader(data_houston_test,batch_size=10,shuffle=True,num_workers=5,pin_memory=True,drop_last=False)

# ------- 3. define model --------
# define the net
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
net = DU_IN(50, 21)


def test():
    ite_num_test = 0
    ite_num4val_test = 0
    running_loss_test = 0.0
    running_tar_loss_test = 0.0
    net.eval()
    for i, data in enumerate(dataSetTest):
        ite_num_test = ite_num_test + 1
        ite_num4val_test = ite_num4val_test + 1
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
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)


        loss2, loss = muti_cel_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        running_loss_test += loss.data.item()
        running_tar_loss_test += loss2.data.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss
        torch.cuda.empty_cache()
        print("=" * 60)
        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] test loss: %3f, tar: %3f " % (
        1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num_test, running_loss_test / ite_num4val_test, running_tar_loss_test / ite_num4val_test))
        print("=" * 60)

def train():

    if torch.cuda.is_available():
        net.to(device=device)

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.0007, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 100 # save the model every 2000 iterations

    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(dataSet):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

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
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)


            loss2, loss = muti_cel_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            if ite_num % save_frq ==0:
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0

        if epoch and epoch % 200 == 0:
            torch.save(net.state_dict(), model_dir + model_name+"_cel_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))


        test()
if __name__ == '__main__':
    train()