import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
oldrec=np.load('carseqrects.npy')
rect = np.array([59, 116, 145, 151], dtype='float64')

rect_1=rect


p=np.zeros(2)

m=np.array([rect])
    
pnminus1=p=np.zeros(2)


for i in range(1, seq.shape[2]):
    print(i)
    I=seq[:,:,i]
    T=seq[:,:,i-1]
    pn=LucasKanade(T,I, rect, threshold, num_iters, pnminus1)
    p+=pn
    pnstar=LucasKanade(seq[:,:,0], I, rect_1, threshold, num_iters, p)
    
   
    if abs(np.sum(pnstar-p-pn))<20:
        rect = rect_1+np.array((pnstar[0], pnstar[1], pnstar[0], pnstar[1]))
        
        pnminus1 = pnstar - p
    else:
    
       
        rect[0]+=pn[0]
        rect[1]+=pn[1]
        rect[2]+=pn[0]
        rect[3]+=pn[1]
       
        pnminus1=pn


    
   
   
    m=np.vstack((m, rect))
        
        
fig, ((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(1,5)


np.save('carseqrects-wcrt.npy', m)


ax1.axis('off')
ax1.imshow(seq[:,:,0], cmap='gray')
rect=m[0]
re1=oldrec[0]
rec = patches.Rectangle((rect[0],rect[1]),rect[2]-rect[0],rect[3]-rect[1],linewidth=1,edgecolor='r',facecolor='none')
orec=patches.Rectangle((re1[0],re1[1]),re1[2]-re1[0],re1[3]-re1[1],linewidth=1,edgecolor='b',facecolor='none')
# Add the patch to the Axes
ax1.add_patch(rec)
ax1.add_patch(orec)


ax2.axis('off')
ax2.imshow(seq[:,:,99], cmap='gray')
rect=m[99]
re1=oldrec[99]
rec = patches.Rectangle((rect[0],rect[1]),rect[2]-rect[0],rect[3]-rect[1],linewidth=1,edgecolor='r',facecolor='none')
orec=patches.Rectangle((re1[0],re1[1]),re1[2]-re1[0],re1[3]-re1[1],linewidth=1,edgecolor='b',facecolor='none')
# Add the patch to the Axes
ax2.add_patch(rec)
ax2.add_patch(orec)

ax3.axis('off')
ax3.imshow(seq[:,:,199], cmap='gray')
rect=m[199]
rec = patches.Rectangle((rect[0],rect[1]),rect[2]-rect[0],rect[3]-rect[1],linewidth=1,edgecolor='r',facecolor='none')
re1=oldrec[199]
orec=patches.Rectangle((re1[0],re1[1]),re1[2]-re1[0],re1[3]-re1[1],linewidth=1,edgecolor='b',facecolor='none')

# Add the patch to the Axes
ax3.add_patch(rec)
ax3.add_patch(orec)

ax4.axis('off')
ax4.imshow(seq[:,:,299],cmap='gray')
rect=m[299]
rec = patches.Rectangle((rect[0],rect[1]),rect[2]-rect[0],rect[3]-rect[1],linewidth=1,edgecolor='r',facecolor='none')
re1=oldrec[299]
orec=patches.Rectangle((re1[0],re1[1]),re1[2]-re1[0],re1[3]-re1[1],linewidth=1,edgecolor='b',facecolor='none')
# Add the patch to the Axes
ax4.add_patch(rec)
ax4.add_patch(orec)


ax5.axis('off')
ax5.imshow(seq[:,:,399], cmap='gray')
rect=m[399]
rec = patches.Rectangle((rect[0],rect[1]),rect[2]-rect[0],rect[3]-rect[1],linewidth=1,edgecolor='r',facecolor='none')
re1=oldrec[399]
orec=patches.Rectangle((re1[0],re1[1]),re1[2]-re1[0],re1[3]-re1[1],linewidth=1,edgecolor='b',facecolor='none')
# Add the patch to the Axes
ax5.add_patch(rec)
ax5.add_patch(orec)

plt.show()
        
        
        
        
        
    
    