'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''


import numpy as np
from submission import eightpoint
from submission import essentialMatrix
from submission import triangulate
from helper import camera2
import cv2
from submission import rodrigues
from submission import invRodrigues

k=np.load("../data/intrinsics.npz")
d=np.load("../data/some_corresp.npz")


f=eightpoint(d['pts1'], d['pts2'], 640)
e=essentialMatrix(f, k['K1'], k['K2'])
r=camera2(e)
r1=np.array([[1,0,0,0],[0,1,0,0], [0,0,1,0]])
c1=k['K1']@r1

pts1=d['pts1']
pts2=d['pts2']
correctM=0
j=0
for i in range(4):
    c2=k['K2']@r[:,:,i]
    p1,e=triangulate(c1, pts1, c2, pts2)
    if np.min(p1[:,2])>0:
        correctM=r[:,:,i]
        j=i

np.savez('q3_3.npz', M=correctM)        

