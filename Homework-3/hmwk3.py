#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 18:51:15 2018

@author: deepayanbhadra
"""
import numpy as np
import sympy as sym
import hmwk2


# Q1(a) This function returns true if grad generates the gradient of f
def check_gradient(f,grad,x0):
        delta = np.random.randn(*x0.shape) # Unpacking operator
        delta = np.linalg.norm(x0, ord=2)* delta/np.linalg.norm(delta, ord=2)  
        rel_err = []
        for i in range(10):
            norm_delx0 = '||delta||/||x0|| = {}\n'.format(np.linalg.norm(delta,
                                                           ord=2)/np.linalg.norm(x0,ord=2))           
            grad_RHS = np.sum(delta*grad(x0))
            grad_LHS = f(x0+delta) - f(x0)        
            grad_cond = (grad_LHS - grad_RHS)/grad_LHS
            diff_delx0 = 'Difference between LHS and RHS = {}'.format(np.absolute(grad_RHS-grad_LHS))
            rel_err.append(grad_cond)
            delta = delta/10.0        
            print (norm_delx0 + diff_delx0,'\n')
        print ('Min relative error is {}'.format(np.amin(np.absolute(rel_err))),'\n')
        return (np.amin(np.absolute(rel_err)) < 1e-06)

# Q2(a) This function produces the gradient of the function 
#        logreg objective from your last homework
        
def logreg_grad(x,D,c):
    z = np.diagflat(c).dot(D).dot(x);
    idxN, idxP = z<0, z>=0 # logical indexing
    y1 = [-1 + np.exp(x)/(1+np.exp(x)) for x in z[idxN]]
    y1 = np.array(y1)
    y2 = [-np.exp(-x)/(1+np.exp(-x)) for x in z[idxP]]
    y2 = np.array(y2)

    y = np.empty(z.shape, dtype='float')
    y[idxN] = y1                           # values for negative indices
    y[idxP] = y2                           # values for positive indices
    
    temp = D.transpose().dot(np.diagflat(c)).dot(y); # grad f(CDx) = D'*C'*f'(CDx)
    return y

# Q3(a) This function returns the smoothed l1 norm
def l1_eps(x,eps):
    x = x.flatten(1)
    f = np.sum(np.sqrt(np.square(x)+np.square(eps)))
    return f

# Q3(b) This function returns the gradient of the smoothed l1 norm
def l1_grad(x,eps):
    f = np.divide(x.flatten(1),np.sqrt(np.square(x.flatten(1))+np.square(eps)))
    f = np.reshape(f,x.shape)
    return f

# Q3(c) 
def tv_objective(x,b,mu,eps):
    f = mu*l1_eps(grad2d(x),eps)+0.5*(np.linalg.norm(x-b))**2
    return f

# Q3(d)
def tv_grad(x,b,mu,eps):
    f = mu*div2d(l1_grad(grad2d(x),eps)) +(x-b);
    return f

# Q4(a) 


if __name__ == "__main__":


    # Q1(b) to test gradient checker
    
    x = sym.MatrixSymbol('x',3,5)
    A = np.random.randn(4,3)
    b = np.random.randn(4,5)
    
    f = lambda x: 0.5*(np.linalg.norm(A.dot(x)-b))**2
    x0 = np.random.randn(3,5)
    grad = lambda x: A.transpose().dot((A.dot(x)-b));
    output = check_gradient(f,grad,x0)
    if output == True:
        print ("Gradient Checker is successful") 
    else: 
        print ("Gradient Checker is not successful") 
        
    input("Press Enter to continue...")
    
    # Q2(b) to test gradient with classification problem
    
    [ D,c ] = create_classification_problem(1000, 10, 5);  
    x = np.random.randn(10,1);
    f = lambda x:logreg_objective(x,D,c)
    grad = lambda x:logreg_grad(x,D,c)
    output = check_gradient(f,grad,x)
    
    # Q3(e) to test gradient using noisy test image
    
    x = np.random.randn(64,64)
    b = np.zeros(x.shape)
    b[16:48,16:48] = 1
    mu = 0.5
    f_obj = lambda x:tv_objective(x,b,mu,np.finfo(float).eps)
    f_grad = lambda x:tv_grad(x,b,mu,np.finfo(float).eps)
    output = check_gradient(f_obj,f_grad,x)
    
    # EOF
