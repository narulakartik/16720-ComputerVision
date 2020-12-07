import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
from sklearn.cluster import KMeans
import random


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
   
    if len(img.shape)==2:
        img=np.dstack((img,img,img))
    
    
    modified_img=skimage.color.rgb2lab(img)

    r_channel=modified_img[:,:,0]
    g_channel=modified_img[:, :,1]
    b_channel=modified_img[:, : ,2]

    
    
    g=[]
    for i in range(len(filter_scales)):
        modified_r_channel=scipy.ndimage.gaussian_filter(r_channel, filter_scales[i])
        modified_g_channel=scipy.ndimage.gaussian_filter(g_channel, filter_scales[i])
        modified_b_channel=scipy.ndimage.gaussian_filter(b_channel, filter_scales[i])
        
        modified_r_dog_X=scipy.ndimage.gaussian_filter(r_channel, filter_scales[i],(0,1))
        modified_g_dog_X=scipy.ndimage.gaussian_filter(g_channel, filter_scales[i], (0,1))
        modified_b_dog_X=scipy.ndimage.gaussian_filter(b_channel, filter_scales[i], (0,1))
    
        modified_r_dog_Y=scipy.ndimage.gaussian_filter(r_channel, filter_scales[i],(1,0))
        modified_g_dog_Y=scipy.ndimage.gaussian_filter(g_channel, filter_scales[i], (1,0))
        modified_b_dog_Y=scipy.ndimage.gaussian_filter(b_channel, filter_scales[i], (1,0))
        
        modified_r_log=scipy.ndimage.gaussian_laplace(r_channel, filter_scales[i])
        modified_g_log=scipy.ndimage.gaussian_laplace(g_channel, filter_scales[i])
        modified_b_log=scipy.ndimage.gaussian_laplace(b_channel, filter_scales[i])
        
        
        
        
        a=np.dstack((modified_r_channel, modified_g_channel, modified_b_channel))
        b=np.dstack((modified_r_dog_X, modified_g_dog_X, modified_b_dog_X))
        c=np.dstack((modified_r_dog_Y, modified_g_dog_Y, modified_b_dog_Y))
        d=np.dstack((modified_r_log, modified_g_log, modified_b_log))
        filter_response=np.dstack((a,b,c,d))
        g.append(filter_response)
        
        
        
    filter_responses=np.dstack(g)
    
    # ----- TODO -----
    return filter_responses
    
def compute_dictionary_one_image(opts, img):
    
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    alpha=opts.alpha
    response=extract_filter_responses(opts, img)
    
    
    
    d=response.shape[0]*response.shape[1]
    response=response.reshape((d,-1))
#response=np.squeeze(response)
    

    alphas=np.random.choice(d, alpha)

    alphaed_response=response[alphas]
   
    
    
    # ----- TODO -----
    return alphaed_response

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    
    # ----- TODO -----
    m=[]
    for i in range(len(train_files)):
        
        img_path = join(opts.data_dir, train_files[i])
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32)/255
        re=compute_dictionary_one_image(opts, img)
        m.append(re)
        
    m=np.array(m)
    n=m.shape[0]*m.shape[1]
    final_response=m.reshape((n,-1))
    
    kmeans=KMeans(n_clusters=K).fit(final_response)
    

        
        
    
    
    
    
            
    
    

    ## example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), kmeans.cluster_centers_)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    
    '''
    response=extract_filter_responses(opts, img)
    
    
    response=response.reshape(response.shape[0]*response.shape[1],-1)
    
    dist=scipy.spatial.distance.cdist(response, dictionary)
    
    visual_words=np.argmin(dist, axis=1)
    visual_words=visual_words.reshape(img.shape[0],img.shape[1])
    return visual_words
    
    
    
    
    
    
    # ----- TODO -----
    



