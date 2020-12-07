import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes=[]
    original = skimage.img_as_float(image)
    sigma = 0.155
    noisy = skimage.util.random_noise(original, var=sigma**2)
    sigma_est = skimage.restoration.estimate_sigma(noisy, multichannel=True, average_sigmas=True)
    a=skimage.restoration.denoise_wavelet(noisy, multichannel=True, rescale_sigma=True)
    image = skimage.color.rgb2gray(a)
    thresh = skimage.filters.threshold_otsu(image)
    bw = skimage.morphology.closing(image < thresh, skimage.morphology.square(3))
    cleared = skimage.segmentation.clear_border(bw)        
    label_image = skimage.measure.label(cleared)
    for region in skimage.measure.regionprops(label_image):
         if region.area >= 200:
        # draw rectangle around segmented coins
             bbox = (region.bbox)
             bboxes.append(bbox)
        
    ##########################
    ##### your code here #####
    ##########################

    bw=(image>thresh).astype(np.float)

    return bboxes, (bw)