import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion



# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.25, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq_air = np.load('../data/aerialseq.npy')
g=[]
for i in range(1,seq_air.shape[2]):
    print(i)
    m=SubtractDominantMotion(seq_air[:,:,i-1], seq_air[:,:,i], threshold, num_iters, tolerance)
    g.append(m)
    


fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1,4)





ax1.axis('off')
im=np.dstack((seq_air[:,:,29], seq_air[:,:,29], seq_air[:,:,29]))
im[:,:,2][g[29]==1]=1
ax1.imshow(im)



ax2.axis('off')
im=np.dstack((seq_air[:,:,59], seq_air[:,:,59], seq_air[:,:,59]))
im[:,:,2][g[59]==1]=1
ax2.imshow(im)




ax3.axis('off')
im=np.dstack((seq_air[:,:,89], seq_air[:,:,89], seq_air[:,:,89]))
im[:,:,2][g[89]==1]=1
ax3.imshow(im)


ax4.axis('off')
im=np.dstack((seq_air[:,:,119], seq_air[:,:,119], seq_air[:,:,119]))
im[:,:,2][g[119]==1]=1
ax4.imshow(im)


# Add the patch to the Axes


plt.show()