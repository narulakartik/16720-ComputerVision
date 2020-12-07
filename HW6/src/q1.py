# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

# Imports
import numpy as np
from skimage.io import imread
from skimage.color import rgb2xyz
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy

from utils import integrateFrankot

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centered on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the sphere in an array of size (3,)

    rad : float
        The radius of the sphere

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the sphere
    """
    cx,cy,cz=center
    r=rad
    #image = None
    image = np.ones((res[1], res[0]))
    xp = np.arange(cx-r,cx+r+pxSize, pxSize)
    yp = np.arange(cy-r,cy+r+pxSize, pxSize)
    b,c =np.where((xp[:,np.newaxis] - cx)**2 + (yp - cy)**2 <= r**2)
    d = xp[b]
    e = yp[c]
    nx = (d-cx)/r
    ny = (e-cy)/r
    
    for i in range(len(b)):
        n=np.array([nx[i], ny[i], 1])    
        image[b[i]][c[i]]=np.dot(n,light)

    
    return image


def loadData(path = "../data/"):

   
    b=[]
    for i in range(7):
        a=plt.imread(path+'input_'+str(i+1)+'.tif')
        a=a.astype('uint16')
        xy= rgb2xyz(a)
        l=xy[:,:,1]
        l=l.flatten()
        b.append(l)
    
    I=np.array(b)
    g= np.load(path+'sources.npy')
    L= np.transpose(g)
    
   # I = None
   # L = None
    s = a.shape

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    a=scipy.sparse.csr_matrix(I)

    b1 = np.linalg.inv(L@L.T)
    b2 = b1@L

    B=b2@a   
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''
    albedos=[]
    for i in range(B.shape[1]):
        a=np.linalg.norm(B[:,i])
        albedos.append(a)
    albedos = np.array(albedos)
    normals = B/albedos
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = albedos.reshape((s[0], s[1]))
    normals=normals.T
    normalIm = normals.reshape((s[0], s[1],3))

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    normals=normals.T
    normals=normals.reshape((s[0], s[1],3))
    G=np.zeros((s[0], s[1]))
    H=np.zeros((s[0], s[1]))

    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            G[i][j]=-normals[i][j][0]/normals[i][j][2]
            H[i][j]=-normals[i][j][1]/normals[i][j][2]
   
   


    surface=integrateFrankot(G,H)

#    surface = None
    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.azim=20
    ax.elev=20
    ax.dist=10
    x=np.arange(1, 432, 1)
    y=np.arange(1, 370, 1)
    X,Y=np.meshgrid(y,x)
    ax.plot_surface(X,Y,surface, cmap='coolwarm')

    


if __name__ == '__main__':

   res= np.array([3840,2160])
   pxSize=7e-4
   cx=0
   cy=0
   cz=0
   r=0.75
   a= np.array([-1, -1, 1])/np.sqrt(3)


   center=np.array([cx, cy, cz])
   im=renderNDotLSphere(center, r,a,pxSize,res)    
    
   I,L,S = loadData()
   sing_values=scipy.linalg.svdvals(I)
   
   B=estimatePseudonormalsCalibrated(I,L)
   
   albedos, normals = estimateAlbedosNormals(B)

   albedoIm, normalIm = displayAlbedosNormals(albedos, normals, S)
   surface=estimateShape(normals, S)
   plotSurface(surface)
   
   
   
   
   
   
   
   
   
   
   
   
   
   

