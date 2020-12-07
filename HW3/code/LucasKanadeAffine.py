import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array] put your implementation here
    """
    
    
    # put your implementation here
    M = np.eye(3)
    
    x=It.shape[0]
    y=It.shape[1]
    X=np.arange(0, x+1)
    Y=np.arange(0, y+1)
    gridx, gridy= np.meshgrid(X,Y)
    
    
    spline=RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    t_spline=RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)

    template=t_spline.ev(gridy, gridx)
    dp=np.array([50,50])
    for i in range(10000):
        while np.sum(dp**2)> threshold:
            Z=np.matmul(M, np.array([gridx, gridy, 1]))
            
            warped_image=spline.ev(Z[1],Z[0])
            
            error=template-warped_image
            
            Ix=spline.ev(Z[1], Z[0], dx=0,dy=1)
            Iy=spline.ev(Z[1], Z[0], dx=1,dy=0)
            
           
            
            A = np.hstack((Ix.reshape(-1,1), Iy.reshape(-1,1)))
            
            xc=Z[0].flatten()
            yc=Z[1].flatten()
            
            
            
            sd=[]

            for i in range(len(xc)):
                J=np.array([[xc[i], yc[i],1,0,0,0],[0,0, 0, xc[i],yc[i] ,1]])
                sdi=(A[i]@J)
                sd.append(sdi)
                
            sd=np.array(sd)
            
            H=np.matmul(sd.T, sd)
            
            temp=np.dot(sd.T, error.flatten())
            dp=np.matmul(np.linalg.inv(H),temp)
            
            M[0]+=np.array([dp[0], dp[1], dp[2]])
            M[1]+=np.array([dp[3], dp[4], dp[5]])
            
            
            
    
    

    return M
