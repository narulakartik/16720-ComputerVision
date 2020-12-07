import numpy as np
import cv2
from matchPics import matchPics
import scipy
from opts import get_opts
import matplotlib.pyplot as plt
from helper import plotMatches

opts=get_opts()
#count=[]
I=cv2.imread('../data/cv_cover.jpg')
#for i in range(1,35):
 #   rot=scipy.ndimage.rotate(I, (i+1)*10)
  #  match, locs1, locs2=matchPics(I, rot, opts)
   # count.append(len(match))	
    
	
#plt.hist(count)

#Display histogram


rot=scipy.ndimage.rotate(I, (10+1)*10)
match, locs1, locs2=matchPics(I, rot, opts)
plotMatches(I, rot, match, locs1, locs2)