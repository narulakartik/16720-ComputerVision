import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """

    # Put your implementation here
    p=p0
    dp=np.array([50,50])
    x=np.arange(rect[0], rect[2]+1, 1)
    y=np.arange(rect[1], rect[3]+1, 1)
   
    gridx, gridy= np.meshgrid(x,y)
    
    spline=RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    t_spline=RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    
    template=t_spline.ev(gridy, gridx)
    
    for i in range(10000):
        while np.sum(dp**2)> threshold:
            
            warp=np.array([[1,0,p[0]],[0,1,p[1]]])  
            
            X=np.matmul(warp, np.array([gridx, gridy, 1]))
            
            warped_image=spline.ev(X[1],X[0])
            
            
            
            error=template-warped_image
            
            Ix=spline.ev(X[1], X[0], dx=0,dy=1)
            Iy=spline.ev(X[1], X[0], dx=1,dy=0)
            
            A = np.hstack((Ix.reshape(-1,1), Iy.reshape(-1,1)))
            
            #jacobian=np.identity(2)
            H = np.matmul(np.transpose(A), A)
            E  =(np.transpose(A)).dot(error.flatten())
            dp= np.linalg.inv(H)@E
            
            
            p=p+dp
            
            
            
            
            
            
                  
                
                
                

       
           
           


    return p








	
 
   










