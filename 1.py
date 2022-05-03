import torch
import torch.nn as nn
m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn((3,2), requires_grad=True)
target = torch.empty(3,1).random_(2)
output = loss(m(input), target)
output.backward()