import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, words):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    K=opts.K
    hist, bins=np.histogram(words, K)    
    hist=hist.astype('float64')
    h=np.linalg.norm(hist,1)
    hist/=h
    
    
    return hist
    
    
    # ----- TODO -----
    

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----
    weights=[]
    for i in range(L):
        if i==0 :
            weights.append(pow(2,-L+1))
        else :
            weights.append(weights[i-1]*2)
    B=wordmap.shape[0]
    H=wordmap.shape[1]
    
    final=np.array([])
    for k in range(L):
            for i in range(pow(2,k)):
                for j in range(pow(2,k)):
                    subset=wordmap[i*B//pow(2,k):(i+1)*B//pow(2,k), j*H//pow(2,k):(j+1)*H//pow(2,k)]   
            
                    hist=get_feature_from_wordmap(opts, subset) 
                    hist*=weights[k]
                    final=np.append(final,hist)            
            
    h=np.linalg.norm(final,1)
    final/=h
    return final.reshape((1, final.shape[0]))
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    visual_map= visual_words.get_visual_words(opts, img, dictionary)
    SPM=get_feature_from_wordmap_SPM(opts, visual_map)
    
    
    # ----- TODO -----
    return SPM

def build_recognition_system(opts, n_worker=4):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    
    img_path = join(opts.data_dir, train_files[0])
    features=get_image_feature(opts, img_path, dictionary)
   
    for i in range(1,len(train_files)):
        img_path = join(opts.data_dir, train_files[i])
    
        SPM_i=get_image_feature(opts, img_path, dictionary)
        features=np.vstack((features, SPM_i))
        
    # ----- TODO -----
    
    
    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
     features=features,
         labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
     )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    return np.sum(np.minimum(word_hist, histograms), axis=1)

    # ----- TODO -----
        
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    conf=np.zeros((8,8))
    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)
    
    for i in range(len(test_files)):
            img_path = join(opts.data_dir, test_files[i])
            img=Image.open(img_path)
            img = np.array(img).astype(np.float32)/255
           
            wordmap=visual_words.get_visual_words(opts, img, dictionary)
            word_hist=get_feature_from_wordmap_SPM(opts, wordmap)
            lf=distance_to_set(word_hist, trained_system['features'])
            predicted_index=np.argmax(lf)
            predicted_class=trained_system['labels'][predicted_index]
            real=test_labels[i]
            conf[real][predicted_class]+=1
                
    accuracy=np.trace(conf)/np.sum(conf)

    return conf, accuracy        
            
                   
                   
                   
                   

    # ----- TODO -----
    
















