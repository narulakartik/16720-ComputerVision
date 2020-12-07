# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:31:19 2020

@author: narul
"""
from os.path import join
from nn import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import numpy as np
from PIL import Image
import skimage.transform
import torch.optim as optim

data_dir='../hw1_2020fall/data'
train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
train_labels= open(join(data_dir, 'train_labels.txt')).read().splitlines() 
train_labels=np.array(train_labels, dtype='int8')
train_data=[]
for i in range(len(train_files)):
        
        img_path = join(data_dir, train_files[i])
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32)/255
        img = skimage.transform.resize(img, (200,200))
        train_data.append(img)
        
        
        
        
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(3,6, 5)
        self.conv2=nn.Conv2d(6,16, 5)
        self.l1=nn.Linear(16*47*47, 64)
        self.l2=nn.Linear(64, 8)
       
    
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

batch_size=32
train_data=np.array(train_data)
train_data=train_data.reshape((-1, 3, 200, 200))
loss = nn.CrossEntropyLoss()
#train_data=torch.from_numpy(train_data)
optimizer = optim.SGD(net.parameters(), lr=0.01)

epochs=50
batches=get_random_batches(train_data, train_labels, batch_size)
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
     
        target=torch.from_numpy(yb)
        target=target.long()
        for i in range(len(target)):
             if target[i].item()==torch.max(out, dim=1)[1][i].item():
                    t_acc+=1
        output = loss(out, target)
        total_loss+=output.item()
        output.backward()
        optimizer.step()
    l.append(total_loss/(1184))
    t_acc/=1184
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


test_data=[]
test_=[]
test_labels= open(join(data_dir, 'test_labels.txt')).read().splitlines()
test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
for i in range(len(test_files)):
        
        img_path = join(data_dir, test_files[i])
        img = Image.open(img_path)
       
        img = np.array(img).astype(np.float32)/255
        
        img = skimage.transform.resize(img, (200,200))
        if len(img.shape)==3:
            test_data.append(img)
            test_.append(test_labels[i])
 
test_=np.array(test_, dtype='int8')

target=torch.from_numpy(test_)
target=target.long()
test_data=np.array(test_data)
test_data=test_data.reshape((-1, 3, 200, 200))
x=torch.from_numpy(test_data)
out=net(x)
output = loss(out, target)
t_acc=0
for i in range(len(target)):
             if target[i].item()==torch.max(out, dim=1)[1][i].item():
                    t_acc+=1


