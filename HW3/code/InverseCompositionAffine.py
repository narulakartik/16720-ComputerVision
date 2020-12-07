import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array]
    """

    # put your implementation here
    M = np.eye(3)
    
    
    T=It
    I=It1
    x=T.shape[0]
    y=T.shape[1]
    X=np.arange(0, x+1)
    Y=np.arange(0, y+1)
    gridx, gridy= np.meshgrid(X,Y)
    spline=RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    t_spline=RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    
    template=t_spline.ev(gridy, gridx)
    Tx=t_spline.ev(gridy, gridx, dx=0,dy=1)
    Ty=t_spline.ev(gridy, gridx, dx=1,dy=0)
    A=np.hstack((Tx.reshape(-1,1), Ty.reshape(-1,1)))
    
    xc=gridx.flatten()
    yc=gridy.flatten()
    sd=[]
    for i in range(len(xc)):
         J=np.array([[xc[i], yc[i],1,0,0,0],[0,0, 0, xc[i],yc[i] ,1]])
         sdi=(A[i]@J)
         sd.append(sdi)
    sd=np.array(sd)
    H=np.matmul(sd.T, sd)   
    dp=np.array([50,50])
    
    for i in range(10000):
        while np.sum(dp**2)> threshold:
              Z=np.matmul(M, np.array([gridx, gridy, 1]))
            
              warped_image=spline.ev(Z[1],Z[0])
              error=warped_image-template
              temp=np.dot(sd.T, error.flatten())
              dp=np.matmul(np.linalg.inv(H),temp)
              
              wdp=np.array([[1+dp[0], dp[1], dp[2]], [dp[3], 1+dp[4], dp[5]], [0,0,1]])
              wdp_i=np.linalg.inv(wdp)
              
              M=M@wdp_i
              
              
            

    return M
