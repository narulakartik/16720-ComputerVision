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
#template_threshold = args.template_threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]


matrix=np.array([rect])
                        
for i in range(1,seq.shape[2]):
    print(i)
    p=LucasKanade(seq[:,:,i-1], seq[:,:,i], rect, threshold, num_iters)    
    rect[0]+=p[0]
    rect[1]+=p[1]
    rect[2]+=p[0]
    rect[3]+=p[1]
    
    matrix=np.vstack((matrix, rect))
        
        
np.save('girlseqrects.npy', matrix)





fig, ((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(1,5)




ax1.axis('off')
ax1.imshow(seq[:,:,0], cmap='gray')
rect=matrix[0]
rec = patches.Rectangle((rect[0],rect[1]),rect[2]-rect[0],rect[3]-rect[1],linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax1.add_patch(rec)

ax2.axis('off')
ax2.imshow(seq[:,:,19], cmap='gray')
rect=matrix[1]
rec = patches.Rectangle((rect[0],rect[1]),rect[2]-rect[0],rect[3]-rect[1],linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax2.add_patch(rec)

ax3.axis('off')
ax3.imshow(seq[:,:,39], cmap='gray')
rect=matrix[2]
rec = patches.Rectangle((rect[0],rect[1]),rect[2]-rect[0],rect[3]-rect[1],linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax3.add_patch(rec)

ax4.axis('off')
ax4.imshow(seq[:,:,59], cmap='gray')
rect=matrix[3]
rec = patches.Rectangle((rect[0],rect[1]),rect[2]-rect[0],rect[3]-rect[1],linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax4.add_patch(rec)


ax5.axis('off')
ax5.imshow(seq[:,:,79], cmap='gray')
rect=matrix[4]
rec = patches.Rectangle((rect[0],rect[1]),rect[2]-rect[0],rect[3]-rect[1],linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax5.add_patch(rec)

plt.show()