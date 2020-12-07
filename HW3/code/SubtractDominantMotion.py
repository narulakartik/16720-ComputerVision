import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.morphology import binary_dilation
from InverseCompositionAffine import InverseCompositionAffine
def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    # put your implementation here
    
    mask = np.ones((image1.shape), dtype=bool)

    #image2-- t+1, the template
    #image1-- t, the image
    M=LucasKanadeAffine(image2, image1, threshold, num_iters)
    #warp image1 to image2 that is t tp t+1
    #M is the warp from image to template, that is from image 2 to image1
    # or image1=m * image2
    x=image2.shape[1]
    y=image2.shape[0]
    X=np.arange(0, x)
    Y=np.arange(0, y)
    gridx, gridy= np.meshgrid(X,Y)
    
    spline=RectBivariateSpline(np.arange(image1.shape[0]), np.arange(image1.shape[1]), image1)
    spline2=RectBivariateSpline(np.arange(image2.shape[0]), np.arange(image2.shape[1]), image2)  
    Z=np.matmul((M), np.array([gridx, gridy, 1]))
    warped_image=spline2.ev(Z[1],Z[0])
    warped_image1=spline.ev(gridy,gridx)    

    
    mask[abs(warped_image-warped_image1)<tolerance]=0        
    mask = binary_dilation(mask)
           
    return mask

















