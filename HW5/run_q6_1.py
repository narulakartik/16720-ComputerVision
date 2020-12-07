# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:35:40 2020

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

x = torch.from_numpy(train_x)

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.l1=nn.Linear(1024, 64)
        self.l2=nn.Linear(64, 36)
       
    
    def forward(self, x):
        o=torch.sigmoid(self.l1(x))
        #s=nn.Softmax(dim=1)
        o=(self.l2(o))
        return o

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
epochs=5
l=[]
a=[]
e=np.arange(epochs)+1
for epoch in range(epochs):
    total_loss=0
    print(epoch)
    t_acc=0
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

