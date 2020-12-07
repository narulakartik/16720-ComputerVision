# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:18:48 2020

@author: narul
"""

import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nn import *
import matplotlib.pyplot as plt
import torch.optim as optim


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
train_y = train_data['train_labels']
valid_x = valid_data['valid_data']
train_x=train_x.reshape((10800, 1, 32,32))
x = torch.from_numpy(train_x)

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(1,6, 5)
        self.conv2=nn.Conv2d(6,16, 5)
        self.l1=nn.Linear(16*5*5, 64)
        self.l2=nn.Linear(64, 36)
       
    
    def forward(self, x):
       
        x= F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    
net=Net()
net=net.double()
#out=net(x)
loss = nn.CrossEntropyLoss()
#y=np.argmax(train_y, axis=1)
#target=torch.from_numpy(y)
#target=target.long()
#output = loss(out, target)
batch_size = 36
optimizer = optim.SGD(net.parameters(), lr=0.01)
batches=get_random_batches(train_x, train_y, batch_size)
lr=0.01
epochs=50
l=[]
a=[]
e=np.arange(epochs)+1
for epoch in range(epochs):
    total_loss=0
    t_acc=0
    print(epoch)
    for xb,yb in batches:
        x=torch.from_numpy(xb)
        optimizer.zero_grad()
        out=net(x)
        y=np.argmax(yb, axis=1)
        target=torch.from_numpy(y)
        target=target.long()
        
        for i in range(len(target)):
             if target[i].item()==torch.max(out, dim=1)[1][i].item():
                    t_acc+=1
                    
        output = loss(out, target)
        total_loss+=output.item()
        
        output.backward()
        optimizer.step()
    l.append(total_loss/len(train_y))
    t_acc/=len(train_y)
    t_acc*=100
    a.append(t_acc)
        
plt.figure(1)        
plt.plot(e,l)
plt.xlabel('epoch')
plt.ylabel('mean cross entropy loss')
plt.show()

plt.figure(2)
plt.plot(e,a)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()


