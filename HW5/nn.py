import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    b=np.zeros(out_size)
  #  W, b = None, None
    W=np.random.uniform(-np.sqrt(6)/np.sqrt(in_size+out_size), np.sqrt(6)/np.sqrt(in_size+out_size), (in_size, out_size))
    ##########################
    ##### your code here #####
    ##########################

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):    
    res = 1/(1+np.exp(-x))

    ##########################
    ##### your code here #####
    ##########################

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    ##########################
    ##### your code here #####
    ##########################
    
    pre_act=X@W+b
    post_act=activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    
    c=np.max(x, axis=1).reshape(x.shape[0],1)
    x=x-c
    r = np.exp(x)
    s=np.sum(r, axis=1).reshape(x.shape[0],1)
    res=r/s

    ##########################
    ##### your code here #####
    ##########################

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    c=-np.log(probs)
    s=np.multiply(y, c)
    loss=np.sum(s)
    f=np.zeros_like(probs)
    for i in range(len(probs)):
         f[i][np.where(probs[i]==np.max(probs[i]))]=1
    
    acc=np.sum(np.multiply(y,f))/len(y)
    ##########################
    ##### your code here #####
    ##########################

    return loss, acc

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res


def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]   

    d1=activation_deriv(post_act)
    grad_W=(np.transpose(X)@(d1*delta))
    grad_b=(d1*delta)[0,:]
    grad_X=np.transpose(W@np.transpose(d1*delta))
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    ##### your code here #####
    ##########################

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    num_batches=int(len(x)/batch_size)
    ##########################
    ##### your code here #####
    ##########################
    for i in range(num_batches):
        a=np.random.choice(range(x.shape[0]), size = batch_size, replace = False)
        batch_x=x[a]
        batch_y=y[a]
        b=(batch_x, batch_y)
        batches.append(b)    
    return batches
