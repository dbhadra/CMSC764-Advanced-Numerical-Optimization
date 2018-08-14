#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 17:09:49 2018

@author: deepayanbhadra
"""
import numpy as np
def buildmat(m,n,condNumber):
    A = np.random.randn(m, n)
    np.linalg.cond(A)
    U, S, V = np.linalg.svd(A)
    S = np.array([[S[j] if i==j else 0 for j in range(n)] for i in range(m)])   
    S[S!=0]= np.linspace(condNumber,1,min(m,n))
    A=U.dot(S).dot(V)
    return A
# For a 3x5 matrix
    
m,n,condNumber = 3,5,2
print("The 3x5 matrix A and the condition no. are\n")
A = buildmat(m,n,condNumber)
print(np.matrix(A),"\n")
print(np.linalg.cond(A),"\n")

# For a 5x4 matrix
    
m,n,condNumber = 5,4,4
print("The 5x4 matrix A and the condition no. are\n")
A = buildmat(m,n,condNumber)
print(np.matrix(A),"\n")
print(np.linalg.cond(A),"\n")