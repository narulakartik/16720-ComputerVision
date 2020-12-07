import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2, opts):
    ratio=opts.ratio
    sigma=opts.sigma
    g1 = cv2.cvtColor(I1 , cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(I2 , cv2.COLOR_BGR2GRAY)

    locs1=corner_detection(g1, sigma)
    locs2=corner_detection(g2, sigma)

    desc1, locs1=computeBrief(g1, locs1)
    desc2, locs2=computeBrief(g2, locs2)

    bmatch=briefMatch(desc1, desc2, ratio)
    
    return bmatch,locs1, locs2
	
    
 
