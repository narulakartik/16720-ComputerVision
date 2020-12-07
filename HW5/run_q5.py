import numpy as np
import scipy.io
from nn2 import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)
params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(1024,32,params,'layer1')
initialize_weights(32,32,params,'layer2')
initialize_weights(32,32,params,'layer3')
initialize_weights(32,1024,params,'output')
w=np.arange(max_iters)+1
keys=[]
for k,v in params.items():
    keys.append(k)

for k in keys:
    params['m_'+k]=np.zeros(params[k].shape)    
l=[]   
# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb, _ in batches:
        h1=forward(xb, params, 'layer1', activation=relu)
        h2=forward(h1, params, 'layer2', activation=relu)
        h3=forward(h2, params, 'layer3', activation=relu)
        out=forward(h3, params, 'output', activation=sigmoid)
        # training loop can be exactly the same as q2!
        loss=np.sum(np.square(out-xb))
        total_loss+=loss
        
        
        d1 = 2*(out - xb)
        d2 = backwards(d1, params, 'output', sigmoid_deriv)
        d3 = backwards(d2, params, 'layer3', relu_deriv)
        d4 = backwards(d3, params, 'layer2', relu_deriv)
        backwards(d4, params, 'layer1', relu_deriv)
        
        
    
        #backwards(d4,params,'layer1', relu_deriv)# your loss is now squared error
        # delta is the d/dx of (x-y)^2
        params['m_Wlayer1']=0.9* params['m_Wlayer1']-learning_rate*params['grad_Wlayer1']
        params['Wlayer1']+=params['m_Wlayer1']
        
        params['m_Wlayer2']=0.9* params['m_Wlayer2']-learning_rate*params['grad_Wlayer2']
        params['Wlayer2']+=params['m_Wlayer2']
        
        params['m_Wlayer3']=0.9* params['m_Wlayer3']-learning_rate*params['grad_Wlayer3']
        params['Wlayer3']+=params['m_Wlayer3']
        
        params['m_Woutput']=0.9* params['m_Woutput']-learning_rate*params['grad_Woutput']
        params['Woutput']+=params['m_Woutput']
        
        params['m_blayer1']=0.9* params['m_blayer1']-learning_rate*params['grad_blayer1']
        params['blayer1']+=params['m_blayer1']
        
        params['m_blayer2']=0.9* params['m_blayer2']-learning_rate*params['grad_blayer2']
        params['blayer2']+=params['m_blayer2']
        
        params['m_blayer3']=0.9* params['m_blayer3']-learning_rate*params['grad_blayer3']
        params['blayer3']+=params['m_blayer3']
        
        params['m_boutput']=0.9* params['m_boutput']-learning_rate*params['grad_boutput']
        params['boutput']+=params['m_boutput']
        
        
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################
    l.append(total_loss/len(train_x))
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
        

# Q5.3.1
import matplotlib.pyplot as plt
plt.plot(w,l)
plt.xlabel('Epoch')
plt.ylabel('Loss per sample')
plt.title('Training loss vs Epoch')
# visualize some results
##########################
##### your code here #####
##########################
i1=valid_x
h1=forward(i1, params, 'layer1', activation=relu)
h2=forward(h1, params, 'layer2', activation=relu)
h3=forward(h2, params, 'layer3', activation=relu)
out=forward(h3, params, 'output', activation=sigmoid)
p500=out[500].reshape((32,32))
im500=valid_x[500].reshape((32,32))



# Q5.3.2
a=0
from skimage.measure import compare_psnr as psnr
for i in range(len(valid_x)) :
    a+=psnr(valid_x[i], out[i])
    
a=a/3600
# evaluate PSNR
##########################
##### your code here #####
##########################
