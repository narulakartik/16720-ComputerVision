'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''


import numpy as np
from submission import epipolarCorrespondence
from submission import triangulate
from submission import eightpoint
from submission import essentialMatrix
import matplotlib.pyplot as plt
from helper import camera2


im1=plt.imread('../data/im1.png')
im2=plt.imread('../data/im2.png')
p1=np.load("../data/templeCoords.npz")
d=np.load("../data/some_corresp.npz")
f=eightpoint(d['pts1'], d['pts2'], 640)
k=np.load("../data/intrinsics.npz")
e=essentialMatrix(f, k['K1'], k['K2'])
r=camera2(e)
r1=np.array([[1,0,0,0],[0,1,0,0], [0,0,1,0]])
c1=k['K1']@r1



pts2=np.zeros((288,2))
pts1=np.zeros((288,2))

for i in range(len(p1['x1'])):
    x,y=epipolarCorrespondence(im1, im2, f, p1['x1'][i][0], p1['y1'][i][0])
    pts2[i]=np.array([x,y])
    pts1[i]=np.array([p1['x1'][i][0],p1['y1'][i][0]])
    
for i in range(4):
    c2=k['K2']@r[:,:,i]
    p1,e=triangulate(c1, pts1, c2, pts2)
    if np.min(p1[:,2])>0:
        M2=r[:,:,i]
        j=i
                      
p,e = triangulate(c1, pts1, k['K2']@M2, pts2)               
fig=plt.figure(figsize=(15,20))

ax = plt.axes(projection="3d")
#ax.view_init(azim=0 , elev=90)
ax.set_xlim3d(np.min(p[:,2]), np.max(p[:,2]))
ax.set_ylim3d(np.min(p[:,1]), np.max(p[:,1]))
ax.set_zlim3d(np.min(p[:,0]), np.max(p[:,0]))
ax.scatter3D(p[:,2], p[:,1], p[:,0], c='r', cmap='gray')
plt.xlabel('z')
plt.ylabel('y')
np.savez('q4_2.npz', F=f, M1=r1, C1=c1, M2=M2, C2=k['K2']@M2)