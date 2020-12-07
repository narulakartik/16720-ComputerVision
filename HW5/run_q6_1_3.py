import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.optim as optim


dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())

torch.manual_seed(43)
val_size = 5000
train_size = len(dataset) - val_size
batch_size=128
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=0, pin_memory=True)

for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(3,6, 5)
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
epochs=30
optimizer = optim.SGD(net.parameters(), lr=0.01)
e=np.arange(30)+1
l,a=[],[]
for epoch in range(epochs):
    total_loss=0
    t_acc=0
    print(epoch)
    for xb,yb in train_loader:
        xb=xb.type('torch.DoubleTensor')
        optimizer.zero_grad()
        out=net(xb)
       
        
        
        for i in range(len(yb)):
             if yb[i].item()==torch.max(out, dim=1)[1][i].item():
                    t_acc+=1
                    
        output = loss(out, yb)
        total_loss+=output.item()       
        output.backward()
        optimizer.step()
    l.append(total_loss/len(train_ds))
    t_acc/=len(train_ds)
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



