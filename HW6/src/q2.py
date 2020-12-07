# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
from matplotlib import pyplot as plt


def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """
    u,w,v = np.linalg.svd(I, full_matrices=False)
    w3=w[0:3]

    u3=u[:,:3]
    v3=v[:3, :]
    sigma= np.diag((w3))

    B=v3
#Bp=np.sqrt(sigma)@v3
    L=u3@sigma
   

    return B, L

if __name__ == "__main__":

    # Put your main code here
   I,L,S = loadData()
   mu=0
   nu=0
   lamb=1
   Bp,Lp=estimatePseudonormalsUncalibrated(I)
   G=np.array([[1,0,0], [0,1,0], [mu, nu, lamb]])
   Bp=(np.transpose(np.linalg.inv(G)))@Bp
   A,N=estimateAlbedosNormals(Bp)
   aIm, Nim = displayAlbedosNormals(A, N, S)
   s=S[0:2]
   N_p=enforceIntegrability(N,s)
   surface=estimateShape(N_p,S)
   plotSurface(surface)


