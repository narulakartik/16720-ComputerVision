import numpy as np
from util import *
import scipy.io
#from nn2 import forward, sigmoid, backwards, initialize_weights, softmax, sigmoid_deriv
#from nn2 import get_random_batches, compute_loss_and_acc
from nn import *

import matplotlib.pyplot as plt
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')


train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y=test_data['test_data'], test_data['test_labels']
max_iters = 50
# pick a batch size, learning rate
batch_size = 36
learning_rate = 1e-2
hidden_size = 64
##########################
##### your code here #####
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}
initialize_weights(1024,64,params,'layer1')
initialize_weights(64,36,params,'output')
from mpl_toolkits.axes_grid1 import ImageGrid
b=ImageGrid(plt.figure(4), 111,nrows_ncols=(8,8))
for i in range(64):
    b[i].imshow(params['Wlayer1'][:,i].reshape(32,32))
# initialize layers here
##########################
##### your code here #####
##########################
l=[]
tl, ta=[],[]
a=[]
e=np.arange(50)+1
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0
    for xb,yb in batches:
        h1=forward(xb,params,name='layer1',activation=sigmoid)
        probs=forward(h1,params,'output',softmax)        
        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc= compute_loss_and_acc(yb, probs)
        total_loss+=loss
        avg_acc+=acc*len(yb)
        # backward
        
        d=probs-yb
        delta2 = backwards(d,params,'output', linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)

        # apply gradient
        params['Wlayer1']-=learning_rate*params['grad_Wlayer1']
        params['blayer1']-=learning_rate*params['grad_blayer1']
        params['Woutput']-=learning_rate*params['grad_Woutput']
        params['boutput']-=learning_rate*params['grad_boutput']
        ##########################
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################
    avg_acc=(avg_acc/len(train_y))*100
    total_loss=total_loss/len(train_y)
    l.append(total_loss)
    a.append(avg_acc)
    h1=forward(valid_x,params,name='layer1',activation=sigmoid)
    probs=forward(h1,params,'output',softmax)
    loss_v, acc_v = compute_loss_and_acc(valid_y, probs)
    tl.append(loss_v/len(valid_y))
    ta.append(acc_v*100)
    
    
    
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))
plt.figure(2)
plt.title("lr=0.1")

plt.plot(e, l, label='training data')
plt.legend()
plt.xlabel('Number of Epoch')
plt.ylabel('Cross Entropy Loss')


plt.plot(e, tl, label='vaidation data')
plt.legend()
plt.show()

plt.figure(3)
plt.title("lr=0.1")

plt.plot(e, a, label='training data')
plt.legend()
plt.xlabel('Number of Epoch')
plt.ylabel('Accuracy')


plt.plot(e, ta, label='vaidation data')
plt.legend()
plt.show()

# run on validation set and report accuracy! should be above 75%
h1=forward(valid_x,params,name='layer1',activation=sigmoid)
probs=forward(h1,params,'output',softmax)
loss, valid_acc = compute_loss_and_acc(valid_y, probs)

##########################
##### your code here #####
##########################

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3


a=ImageGrid(plt.figure(4), 111,nrows_ncols=(8,8))
for i in range(64):
    a[i].imshow(saved_params['Wlayer1'][:,i].reshape(32,32))
    


# visualize weights here
##########################
##### your code here #####
##########################

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

h=forward(test_x, saved_params, 'layer1', activation=sigmoid)
probs=forward(h, saved_params,'output', softmax)
lo, ac= compute_loss_and_acc(test_y, probs)
y_pred=np.argmax(probs, axis=1)
y_t=np.argmax(test_y, axis=1)
for i in range(len(y_t)):
    
    confusion_matrix[y_t[i]][y_pred[i]]+=1

# compute comfusion matrix here
##########################
##### your code here #####
##########################

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()