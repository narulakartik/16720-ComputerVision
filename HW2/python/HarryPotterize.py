import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from planarH import computeH_ransac
from planarH import computeH_norm
from planarH import computeH
from matchPics import matchPics
from helper import plotMatches
import matplotlib.pyplot as plt
from scipy.spatial import distance
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from planarH import compositeH
#Import necessary functions



#Write script for Q2.2.4
opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
img= cv2.imread('../data/hp_cover.jpg')

bmatch, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
la1=np.array([locs1[:,1] ,locs1[:,0]])
la2=np.array([locs2[:,1], locs2[:,0]])
la1=np.transpose(la1)
la2=np.transpose(la2)
l1=la1[bmatch[:,0]]
l2=la2[bmatch[:,1]]





image_size = (cv_desk.shape[1], cv_desk.shape[0])




    




b,i=computeH_ransac(l1,l2,opts)
b=np.linalg.inv(b)



dim=(cv_cover.shape[1], cv_cover.shape[0])
img=cv2.resize(img,dim,interpolation = cv2.INTER_AREA)










final=compositeH(b, img, cv_desk)

final=cv2.cvtColor(final, cv2.COLOR_BGR2RGB) 


