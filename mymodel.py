'''
函数说明: 
Author: hongqing
Date: 2021-08-04 14:23:54
LastEditTime: 2021-08-04 15:23:25
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
numoffinger=21
class Net(nn.Module):
    def __init__(self,type=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(numoffinger-1, 255)
        self.fc2 = nn.Linear(255, 255)
        self.fc3 = nn.Linear(255, 3)
        if(type==1):
            self.fc1.weight.data.normal_(0, 3)   # initialization
            self.fc2.weight.data.normal_(0, 3)
            self.fc3.weight.data.normal_(0, 3)
        if(type==0):
            self.fc1.weight.data.zero_()   # initialization
            self.fc2.weight.data.zero_()
            self.fc3.weight.data.zero_()
        if(type==2):
            self.fc1.weight.data.random_(1,2)   # initialization
            self.fc2.weight.data.random_(1,2)
            self.fc3.weight.data.random_(1,2)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x