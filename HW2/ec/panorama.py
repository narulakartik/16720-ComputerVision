# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:01:44 2020

@author: narul
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from opts import get_opts
from planarH import computeH_ransac
from matchPics import matchPics

opts=get_opts()
pano_left=cv2.imread("../data/P2_left.jpeg")
pano_right=cv2.imread("../data/P2_right.jpeg")

bmatch, locs1, locs2 = matchPics(pano_left, pano_right, opts)
la1=np.array([locs1[:,1] ,locs1[:,0]])
la2=np.array([locs2[:,1], locs2[:,0]])
la1=np.transpose(la1)
la2=np.transpose(la2)
l1=la1[bmatch[:,0]]
l2=la2[bmatch[:,1]]




h,i=computeH_ransac(l1,l2, opts)

width=pano_left.shape[1]+pano_right.shape[1]

height=pano_left.shape[0]+pano_right.shape[0]

warped=cv2.warpPerspective(pano_right, (h) , dsize=(width,height))

warped[0:pano_left.shape[0], 0:pano_left.shape[1]] = pano_left


warped=cv2.cvtColor(warped, cv2.COLOR_BGR2RGB) 
pl=cv2.cvtColor(pano_left, cv2.COLOR_BGR2RGB) 
p2=cv2.cvtColor(pano_right, cv2.COLOR_BGR2RGB) 
w=warped[0:1600, 0:1700]
plt.imshow(w)

