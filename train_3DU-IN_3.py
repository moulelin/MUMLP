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
from metrics import AverageMeter,Evaluator
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
import pdb
import numpy as np
import glob
import os
from get_data.loaddata import getHouston2018


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
net = UNet(50, 20) # 考虑不要背景的数据，背景全是0，只会影像最后的结果，重新考虑采样方式


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
        d0 = net(inputs_v)


        loss = muti_cel_loss_fusion(d0, labels_v)

        loss.backward()
        running_loss_test += loss.data.item()


        # del temporary outputs and loss
        del d0

        torch.cuda.empty_cache()
        print("=" * 60)
        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] test loss: %3f" % (
        1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num_test, running_loss_test / ite_num4val_test))
        print("=" * 60)

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
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 100 # save the model every 2000 iterations
    net.train()
    for epoch in range(0, epoch_num):
        loss_print = AverageMeter()
        OA_print = AverageMeter()
        AA_print = AverageMeter()
        evaluator = Evaluator(20) # 去掉背景数据，这些都是毫无意义的

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
            d0 = net(inputs_v)


            loss = muti_cel_loss_fusion(d0,labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()


            # del temporary outputs and loss

            loss_print.add(running_loss,inputs.size(0))

            evaluator.add_batch(labels, d0)

            del d0
            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val))
            if ite_num % save_frq ==0:
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0
        scheduler.step()
        if epoch and epoch % 200 == 0:
            torch.save(net.state_dict(), model_dir + model_name+"_cel_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))


        test()
if __name__ == '__main__':
    train()