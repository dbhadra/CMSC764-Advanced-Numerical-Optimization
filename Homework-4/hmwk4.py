#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:30:54 2018

@author: deepayanbhadra
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy as sp
import numpy.linalg as nla
from math import pi
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.cluster import KMeans
import hmwk3
from hmwk2 import grad2d, div2d


# Q1(c) Richardson Iteration 

def richardson(A,b,x,t):
    resids = 1
    all_res = []
    while resids > 10e-6:
        x = x+t*(b-A(x))
        resids = nla.norm(b-A(x),'fro')
        all_res = np.append(all_res,resids)
    return all_res,x

# Q1(d) Conjugate gradient 

def conjgrad(A,b,x):
    tol = 10e-6;
    xk = x; rk = b-A(xk); pk = rk # Initial values 
    res = [];
    while nla.norm(rk)>tol:
        grad = A(pk);
        alpk = np.dot(rk.flatten(),rk.flatten())/(np.dot(pk.flatten(),
                                                     grad.flatten()))
        xk = xk+alpk*pk;
        rkk = rk-alpk*grad;
        betak = np.dot(rkk.flatten(),rkk.flatten())/(np.dot(rk.flatten(),
                                                           rk.flatten()))
        pk = rkk+betak*pk;
        rk = rkk;
        res = np.append(res,nla.norm(rk,'fro')) 
        # Storing all the residuals 
        
    return x,resids

# Q1(f) to compute exact solutions using FFT

def l2denoise(b,mu):
    kernel = np.zeros((b.shape[0],b.shape[1]))
    kernel[0,0] = 1
    kernel[0,1] = -1
    Dx = np.fft.fft2(kernel)
    kernel = np.zeros((b.shape[0],b.shape[1]))
    kernel[0,0] = 1
    kernel[1,0] = -1
    Dy = np.fft.fft2(kernel)
    dd = np.divide(1,(mu*(np.conj(Dx)*Dx+np.conj(Dy)*Dy)+1))
    x = np.real(np.fft.ifft2(dd*np.fft.fft2(b)))
    return x

 # Q2(a) Two-moons dataset

def make_moons(n):
    """Create a 'two moons' dataset with n feature vectors, 
        and 2 features per vector."""
        
    assert n%2==0, 'n must be even'
    # create upper moon
    theta = np.linspace(-pi / 2, pi / 2, n//2)
    # create lower moon
    x = np.r_[np.sin(theta) - pi / 4, np.sin(theta)]
    y = np.r_[np.cos(theta), -np.cos(theta) + .5]
    data = np.c_[x, y]
    # Add some noise
    data = data + 0.03 * np.random.standard_normal(data.shape)

    # create labels
    labels = np.r_[np.ones((n//2, 1)), -np.ones((n//2, 1))]
    labels = labels.ravel().astype(np.int32)

    return data,labels


if __name__ == "__main__":

    
    # Q1(b): Quadratic Image Denoising Model
    img= mpimg.imread('lena512.bmp')
    img = img.astype(float)
    b = img/max(img.flatten()) # Scaling to [0,1]
    mu = 2
    x0 = np.random.randn(*b.shape)
    f = lambda x:mu*div2d(grad2d(x))+x-b
    output = f(x0) # Evaluating A at x0 
        
    # Q1(c) Richardson Iteration 
    
    t = 0.05    
    all_res,x = richardson(f,b,x0,t)        
    plt.xlabel('# of iterations')
    plt.ylabel('Residual norm')
    plt.plot(all_res)
    print("# of iterations to convergence is",all_res.size)
    plt.imshow(b) # Noisy image
    plt.imshow(x.astype(float)) # De-noised image
    
    # Q1(d) Conjugate gradient 
    
    [x,res] = conjgrad(f,b,x0)
    plt.xlabel('# of iterations')
    plt.ylabel('Residual norm')
    plt.plot(all_res)
    print("# of iterations to convergence is",all_res.size)
    plt.imshow(b) # Noisy image
    plt.imshow(x.astype(float)) # De-noised image
    
    
    # Q1(f) to compute exact solutions using FFT    
    print('The norm of the gradient of the objective function is')
    nla.norm(f(l2denoise(b,mu)))
    
    # Q2(a) Building a dataset for Spectral Clustering
    
    data,l = make_moons(100)
    plt.scatter(data[l>0,0],data[l>0,1],c='r')
    plt.scatter(data[l<0,0],data[l<0,1],c='b')
    plt.show()    
    S = np.zeros((100,100))
    sig = 0.09
    B = pdist(data)
    C = squareform(B)
    S = np.exp(-C/sig)
    
    # Q2(b) to compute the diagonal normalization matrix
    S_sum = np.sum(S,axis=1,keepdims = True)
    D = np.diagflat(S_sum)
    temp = np.power(S_sum,-0.5)
    S_hat = np.diagflat(temp).dot(S).dot(np.diagflat(temp))
    
    # Q2(c) to compute the EVD of normalized S. 
    E,V = nla.eig(S_hat)
    idx = E.argsort()[::-1]
    E = E[idx]
    V = V[:,idx]
    
    U2 = np.stack([V[:,1],V[:,2]],axis=1)
    plt.scatter(U2[l==1,0],U2[l==1,1],c='r')
    plt.scatter(U2[l==-1,0],U2[l==-1,1],c='b')
    plt.show()
    
    # Q2(d) k-means clustering
    
    kmeans = KMeans(n_clusters = 2)
    y_kmeans = kmeans.fit_predict(U2)
    plt.scatter(U2[y_kmeans == 0, 0], U2[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(U2[y_kmeans == 1, 0], U2[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
    
    
    # Q3(a) Nystrom approximation of S
    
    data,l = make_moons(100000)
    sig = 0.09
    idx = np.random.choice(100000,size = 200, replace=False) # Random Sampling of 200 columns 
    idx_c = np.setdiff1d(np.arange(100000),idx)
    temp1 = cdist(data,data[idx,:]) # Pairwise distance between two sets of observations
    C = np.exp(-temp1/sig)
    W = C[idx,:]
    
    # Q3(b)
    
    Z = C[200:,:]
    W_m = np.sum(W,axis=1,keepdims = True)+ np.sum(np.transpose(Z),
                                                     axis=1,keepdims = True) 
    M_e = np.sum(Z,axis=1,keepdims = True)+ np.transpose(np.sum(Z,
                                                     axis=0,keepdims = True)
                                                    .dot(nla.inv(W))
                                                    .dot(np.transpose(Z))) 
    diagD = np.concatenate([W_m,M_e],axis=0)
    temp = np.multiply(np.repeat(diagD,200,axis=1),C)
    C_n = np.dot(temp,np.diagflat(diagD[idx]))
    W_n = C_n[idx,:]
    M_n = C_n[idx_c,:]
    
    
    # Q3(c)
    
    W_hat = W_n + sp.linalg.sqrtm(nla.inv(W_n)).dot(np.transpose(M_n)).dot(M_n).dot(sp.linalg.sqrtm(nla.inv(W_n))) 
    # Orthogonalization matrix 
    
    D_w,U_w = nla.eig(W_hat)
    D_w,U_w = np.real(D_w), np.real(U_w) 
    
    # Eigen-decomposition W_hat = V*D*V' 
    
    U = np.matmul(C_n,np.matmul(sp.linalg.sqrtm(nla.inv(W_n)),U_w))*np.power(np.expand_dims(D_w,axis=-1),-0.5).T
    
    idx = np.argsort(D_w)[::-1]
    U_w = U_w[idx]
    U = U[:,idx]
    
    
    U = U/np.expand_dims(U[:,0],axis=-1)
    U2 = np.expand_dims(U[:,1],axis=-1)
    plt.scatter(U2[l==1],U2[l==1],c='r')
    plt.scatter(U2[l==-1],U2[l==-1],c='b')
    plt.show()
    # Approximate eigen-vectors
    
    
    # Q3(d) k-means clustering
    
    kmeans = KMeans(n_clusters = 2)
    U2 = U[:,-3:-1]
    y_kmeans = kmeans.fit_predict(U2)
    plt.scatter(U2[y_kmeans == 0, 0], U2[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(U2[y_kmeans == 1, 0], U2[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
    
    
    
    
