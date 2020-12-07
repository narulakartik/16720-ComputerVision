"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import util
import cv2
from scipy.ndimage import gaussian_filter
from math import sqrt
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    
    pts1=pts1/M
    pts2=pts2/M
    
    matrix=np.zeros((pts1.shape[0], 9))
    
    for i in range(pts1.shape[0]):
        
        matrix[i]=np.array([pts2[i][0]*pts1[i][0], pts2[i][0]*pts1[i][1], pts2[i][0], pts2[i][1]*pts1[i][0], pts2[i][1]*pts1[i][1], pts2[i][1], pts1[i][0], pts1[i][1], 1])
    u, v, w= np.linalg.svd(matrix)
    f=w[8,:]
    f=f.reshape((3,3))
    f=util._singularize(f)
    f=util.refineF(f, pts1, pts2)
    
    t = np.diag((1/M, 1/M, 1))
    f = np.transpose(t)@f@t
    
    return f

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E=np.transpose(K2)@F@K1
    return  E 
    
    


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    points=np.zeros((pts1.shape[0], 4))
    projected_points1=np.zeros((pts1.shape[0],2))  
    projected_points2=np.zeros((pts1.shape[0],2))  
    for j in range(pts1.shape[0]):
        a=np.zeros((4,4))
        a[0]=pts1[j][0]*C1[2]-C1[0]
        a[1]=pts1[j][1]*C1[2]-C1[1]
        a[2]=pts2[j][0]*C2[2]-C2[0]
        a[3]=pts2[j][1]*C2[2]-C2[1]    
        u,v,w=np.linalg.svd(a)        
        p=w[3]
        p=p/p[3]
        points[j]=p
    projected_points1=np.transpose(C1@np.transpose(points))
    projected_points2=np.transpose(C2@np.transpose(points))
    projected_points1[:,0]/=projected_points1[:,2]
    projected_points1[:,1]/=projected_points1[:,2]
    projected_points1[:,2]/=projected_points1[:,2]
    
    projected_points2[:,0]/=projected_points2[:,2]
    projected_points2[:,1]/=projected_points2[:,2]
    projected_points2[:,2]/=projected_points2[:,2]
    projected_points1=projected_points1[:,:2]
    projected_points2=projected_points2[:,:2]
    error=(np.linalg.norm((projected_points1-pts1),2))**2+(np.linalg.norm((projected_points2-pts2),2))**2
    return points[:,:3],error

        





'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    p=np.array([x1,y1,1])
    g1=cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2=cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)   
    
    xw=np.linspace(x1-10, x1+10, 21)
    yw=np.linspace(y1-10, y1+10, 21)

    xp, yp= np.meshgrid(xw, yw)
    xp=xp.astype('int')
    yp=yp.astype('int')
    patch1=g1[yp, xp]
    line=F@p
    
    ye=im1.shape[0]
    xs=-(line[2])/line[0]
    xe=-(line[2]+line[1]*ye)/line[0]
    error=[]
    xm=int(((xs+xe)/2))
    x2=np.linspace(xm-10, xm+10, 21)
    x2=x2.astype('int')
    

    for i in range(10, ye-10):
        y2=np.linspace(i-10, i+10, 21)
        y2=y2.astype('int')
        xp2, yp2= np.meshgrid(x2, y2)  
        patch2=g2[yp2, xp2]
        e=np.linalg.norm(gaussian_filter((patch2-patch1),0.05), 2)
        error.append(e)
        
    
    d=np.argmin(error)
    ymatch=d
    xmatch=xm
        
    return xmatch, ymatch
'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=200, tol=0.8):
    # Replace pass by your implementation
    inlier_=[]
    k=[]
    F=[]
    for i in range(nIters):
      #  pts1=pts1[ip]
        inliers=np.zeros(len(pts1))
    
        sel_points=np.random.choice(len(pts1),8)
        p1=pts1[sel_points]
        p2=pts2[sel_points]
        f=eightpoint(p1, p2, M)
        ip=[]
        for i in range(len(pts1)):
           p= np.array([pts1[i][0], pts1[i][1], 1])   
           fpl= f@p #fpprime
           pr=np.transpose((np.array([pts2[i][1], pts2[i][0], 1])))
           num=pr@fpl
           den=sqrt(fpl[0]**2+ fpl[1]**2)
           dist=num/den
           if dist < tol:
                ip.append(i)   
                inliers[i]=1
        a=len(ip)     
        inlier_.append(inliers)
        k.append(a)
        F.append(f)
    
    return F[np.argmax(k)], inlier_[np.argmax(k)] 
        
'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    theta=np.linalg.norm(r)    
    r=r/theta
    a=r[0]
    b=r[1]
    c=r[2]
    k=np.array([[0,-c, b], [c, 0, -a],[-b, a,0]])
    k2 = np.array([
            [a*a, a*b, a*c],
            [b*a, b*b, b*c],
            [c*a, c*b, c*c]]
        )
    R=np.cos(theta)*np.identity(3)+np.sin(theta)*(k)+(1-np.cos(theta))*(k2)
    return R    

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    t=np.trace(R)
    angle=np.arccos((t-1)/2)
    a1=R[2][1]-R[1][2]
    a2=R[0][2]-R[2][0]
    a3=R[1][0]-R[0][1]
    r=(angle/(2*np.sin(angle)))*np.array([a1,a2,a3])
    return r
    
    
    
    

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    
    pass
    
    
    
    
    
    




'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass



    



