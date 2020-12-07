import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

#import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    
    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    c=[]
    mean_h=0
    for box in bboxes:
        minr, minc, maxr, maxc = box
        yc=(maxr+minr)/2
        xc=(maxc+minc)/2
        h=(maxr-minr)
        w=(maxc-minc)
        mean_h+=h
        p=np.array([yc, xc, h, w])
        c.append(p) 
    c.sort(key=lambda x:(x[0], x[1]))
    rows, row=[],[]
    mean_h/=len(bboxes)
    t=c[0][0]
    for p in c:
        if p[0]<t+mean_h:
            row.append(p)
            continue
        row.sort(key=lambda x:x[1])
        rows.append(row)
        t=p[0]
        row=[p]
    row.sort(key=lambda x:x[1])        
    rows.append(row) 
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    image_rows=[]
    
    for row in rows:
        subimages=[]
        for box in row:
            yc, xc , h, w = box
            subimage=bw[int(yc-h/2):int(yc+h/2), int(xc-w/2): int(xc+h/2)]
            
            if h > w:
                h_pad = h/20
                w_pad = (h-w)/2+h_pad
            elif h < w:
                w_pad = w/20
                h_pad = (w-h)/2+w_pad
            subimage = np.pad(subimage, ((int(h_pad), int(h_pad)), (int(w_pad), int(w_pad))), 'constant', constant_values=(1, 1))
            subimage = skimage.transform.resize(subimage, (32,32)) 
            subimage = skimage.morphology.erosion(subimage, np.array([[0, 2, 0], [2, 2, 2], [0, 2, 0]]))
            subimage = np.transpose(subimage)            
            subimage = subimage.flatten()
            subimages.append(subimage)
        image_rows.append(np.array(subimages))
 #   for bbox in bboxes:
  #      minr, minc, maxr, maxc = bbox
   #     im=bw[minr:maxr, minc:maxc]      
   #     w=maxc-minc
   #     h=maxr-minr
   #     
   #     if h > w:
   #             h_pad = h/20
   #             w_pad = (h-w)/2+h_pad
   #     elif h < w:
    #            w_pad = w/20
    #            h_pad = (w-h)/2+w_pad
   #     im = np.pad(im, ((int(h_pad), int(h_pad)), (int(w_pad), int(w_pad))), 'constant', constant_values=(1, 1))
   #     im = skimage.transform.resize(im, (32,32)) 
       
    #    im=np.transpose(im)
    #    im=im.flatten()
    #    im=im.reshape((1,-1))       
    #    subimages.append(im)

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    for image_row in image_rows:
        h=forward(image_row, params, 'layer1')
        p=forward(h, params, 'output', softmax)
        d=np.argmax(p, axis=1)
        s=''
        for i in range(len(p)):
            s+=(letters[d[i]])
        print(s)
     #   pass
    ##########################
    ##### your code here #####
    ##########################
    