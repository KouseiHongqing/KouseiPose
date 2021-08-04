'''
函数说明: 
Author: hongqing
Date: 2021-08-03 17:32:49
LastEditTime: 2021-08-04 15:54:02
'''
from torch.autograd import Variable
import cv2
import mediapipe as mp
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch = 400000
savepath="E://pose//traindata//"
numoffinger = 21

def transData(data):
    landmarks = data.landmark
    initx = data.landmark[0].x
    inity = data.landmark[0].y
    res = []
    for landmark in landmarks:
        tmp = []
        tmp.append(landmark.x-initx)
        tmp.append(landmark.y-inity)
        res.append(tmp)
    return res

def getTransData(savepath):
    trainfiles = os.listdir(savepath)
    trainDict = {}
    for trainfile in trainfiles:
        picklefile = savepath + trainfile
        cate = []
        with open(picklefile, 'rb') as file:   
            train = pickle.load(file)
            for tmp in train:
                cate.append(transData(tmp[0]))
            trainDict[trainfile] = cate
    return trainDict
def formatData(trainDict):
    train_data = {}
    for i in trainDict:
        #i = data['1']
        tmp = np.array([])
        for ndata in trainDict[i]:
            #ndata = data['1'][0]
            ndata = np.array(ndata)[1:].ravel()
            tmp = np.r_[tmp,ndata]
        tmp = tmp.reshape(-1,(numoffinger-1)*2)
        train_data[i] = tmp
    return train_data
# def formatData(trainDict):
#     train_data = {}
#     for i in trainDict:
#         tmp = np.array([])
#         for ndata in trainDict[i]:
#             ndata = np.array(ndata)[1:]
#             ndata = ndata[:,1]/ndata[:,0]
#             tmp = np.r_[tmp,ndata]
#         tmp = tmp.reshape(-1,numoffinger-1)
#         train_data[i] = tmp
#     return train_data

class Net(nn.Module):
    def __init__(self,type=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear((numoffinger-1)*2, 255)
        self.fc2 = nn.Linear(255, 510)
        self.fc3 = nn.Linear(510, 510)
        self.fc4 = nn.Linear(510, 3)
        self.bn = nn.BatchNorm1d((numoffinger-1)*2, momentum=0.01)
    def forward(self, x):
        x = self.bn(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x
def mixData(transform_data):
    res = np.array([])
    label = 0
    for i in transform_data:
        for j in transform_data[i]:
            res = np.append(res,j)
            res = np.append(res,label)
        label+=1
    res = res.reshape(-1,(numoffinger-1)*2+1)
    return res

learning_rate = 0.001
net = Net(4).to(device)
net.eval()
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def getData():
    #获取数据
    data = getTransData(savepath)
    #转换数据
    transform_data = formatData(data)
    #混合标签
    mixed_data = mixData(transform_data)
    return mixed_data

# def load():
#     _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neek)
#     pretrained_dict = torch.load(yolov4conv137weight)

#     model_dict = _model.state_dict()
#     # 1. filter out unnecessary keys
#     pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
#     # 2. overwrite entries in the existing state dict
#     model_dict.update(pretrained_dict)
#     _model.load_state_dict(model_dict)

mixed_data = getData()
train_X=mixed_data
# train_X = mixed_data[:,:-1]
# train_Y = mixed_data[:,-1][:,np.newaxis]
trainloader = torch.utils.data.DataLoader(train_X, batch_size=32,shuffle=True)
#  num_workers=1
T = 0
N = 0
for i in range(1,epoch+1):
    for _, data in enumerate(trainloader):
        loscount = 0
        outputs = net(data[:,:-1].float())
        y = data[:,-1].long()

        train_predict = torch.max(outputs, 1)[1] 
        t =torch.sum(train_predict==y).data
        T += t
        N += data.shape[0] - t

        loss = criterion(outputs, y)
        loscount += loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch:{},loss:{}'.format(i,loscount))
    if(i%1000==0):
        torch.save(net.state_dict(), 'lastweight.ckpt')
        print('acc={},saved:{}'.format(T/(T+N),i))
        T = 0
        N = 0
        
        