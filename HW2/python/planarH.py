import numpy as np
import cv2
from scipy.spatial import distance


def computeH(x1, x2):
    matrix=np.zeros((2*x1.shape[0], 9))
    for i in range(len(x1)):
        matrix[2*i]=np.array([[x2[i][0],x2[i][1], 1, 0, 0, 0,-x2[i][0]*x1[i][0],-x2[i][1]*x1[i][0],-x1[i][0]]])
        matrix[2*i+1]=np.array([[0,0,0,x2[i][0],x2[i][1],1,-x2[i][0]*x1[i][1],-x2[i][1]*x1[i][1],-x1[i][1]]])
        
      
    u, s, v=np.linalg.svd(matrix)
    h=v[8,:]/v[8,8]
    h=h.reshape((3,3))
    return (h)
    
    
def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
    c1=np.sum(x1, axis=0)/x1.shape[0]
    c2=np.sum(x2, axis=0)/x2.shape[0]

	#Shift the origin of the points to the centroid

    x1_=x1-c1
    x2_=x2-c2
	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    
    x1t=1.414*x1_/np.amax(np.linalg.norm(x1_,2, axis=1))
    x2t=1.414*x1_/np.amax(np.linalg.norm(x2_,2, axis=1))
	#Similarity transform 1
    
    e1=(1.414/np.amax(np.linalg.norm(x1_,2, axis=1)))
    t1=np.array([[e1,0,-c1[0]*e1], [0,e1,-c1[1]*e1], [0,0,1]]) 
	#Similarity transform 2
    e2=(1.414/np.amax(np.linalg.norm(x2_,2, axis=1)))
    t2=np.array([[e2,0,-c2[0]*e2], [0,e2,-c2[1]*e2], [0,0,1]])     
    print(t1)
    print(t2)

	#Compute homography
    H=computeH(x2t,x1t)

	#Denormalization
	
    H2to1 = np.linalg.solve(t1, np.linalg.inv(H).dot(t2))


    return (H2to1)




def computeH_ransac(loc1, loc2, opts):
    max_iters = opts.max_iters 
    inlier_tol = opts.inlier_tol 
    length=len(loc1)
    
   
    inliers_=[]
    loc1_homo=np.ones((3,length))
    loc2_homo=np.ones((3,length))
    counts=[]
    
    homography=[]
    loc1_homo[:2,]=np.transpose(loc1)
    loc2_homo[:2,]=np.transpose(loc2)
    
    for i in range(max_iters):
        randp=np.random.choice(length,4)
        s1=loc1[randp]
        s2=loc2[randp]
        
        inlier=np.zeros((length))
        H2to1=computeH(s1,s2)      
        computed=H2to1@loc2_homo
        diff=np.sum(np.square(loc1_homo - (computed/computed[2,:])), axis=0)
        c=0
        for i in range(len(diff)):
            if diff[i] < inlier_tol**2:
                inlier[i]=1
                c+=1
    
                
          
        counts.append(c)
        homography.append(H2to1)
        inliers_.append(inlier)
    index=counts.index(max(counts))
    
    bestH2to1=homography[index]
    inliers=inliers_[index]
    print(max(counts))
    return (bestH2to1), inliers



def compositeH(H2to1, template, img):
    image_size=(img.shape[1], img.shape[0])
    a=cv2.warpPerspective(template, (H2to1) , dsize=image_size)
    r,g=cv2.threshold(a,0,255,cv2.THRESH_BINARY)
    n=cv2.add(img,g)
    hp_cover_warp=cv2.warpPerspective(template, (H2to1) , dsize=image_size)
    final=np.add(n, hp_cover_warp)
    
    
    return final
	
    
    


