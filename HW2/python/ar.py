import numpy as np
import cv2
from loadVid import loadVid
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
from opts import get_opts

#Import necessary functions

opts = get_opts()
cv_cover = cv2.imread('../data/cv_cover.jpg')

book=loadVid('../data/book.mov')
kpanda=loadVid('../data/ar_source.mov')

frames=[]

for f in range(len(kpanda)):
    bmatch, locs1, locs2 = matchPics(cv_cover, book[f], opts)
    la1=np.array([locs1[:,1] ,locs1[:,0]])
    la2=np.array([locs2[:,1], locs2[:,0]])
    la1=np.transpose(la1)
    la2=np.transpose(la2)
    l1=la1[bmatch[:,0]]
    l2=la2[bmatch[:,1]]
  
    b,i=computeH_ransac(l1,l2,opts)
    b=np.linalg.inv(b)
    dim=(cv_cover.shape[1], cv_cover.shape[0])
    img=kpanda[f][180-56:180+56,320-100:320+100]
    img=cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    frames.append(compositeH(b, img, book[f]))
 
#img_array=[]
#for i in frames:
 #   img = cv2.imread(i)
  #  height, width, layers = img.shape
   # size = (width,height)
   # img_array.append(img)
 
size=(frames[1].shape[1], frames[1].shape[0])
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(frames)):
    out.write(frames[i])

    

#Write script for Q3.1
