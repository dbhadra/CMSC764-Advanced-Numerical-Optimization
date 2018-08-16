#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:11:31 2018

@author: deepayanbhadra
"""

import numpy as np
import numpy.linalg as nla
import matplotlib.pyplot  as plt
from hmwk2 import logreg_objective,create_classification_problem
from hmwk3 import logreg_grad

def grad_descent(f,grad,x0):
    tol = 10e-6
    delta = np.random.randn(*x0.shape)
    delta = delta/nla.norm(delta,ord='fro',keepdims=True)
    y = x0+delta
    L= nla.norm(grad(x0)-grad(y))/nla.norm(x0-y)
    tau = 2/L
    xk = x0
    res = nla.norm(grad(x0))
    alp, iterations= 0.1,1
    xk1 = 10*xk
    while nla.norm(grad(xk1)) > nla.norm(grad(x0))*tol:
        xk1 = xk-tau*grad(xk)
        while f(xk1) > f(xk)+alp*(xk1-xk)*grad(xk):
            tau = tau*0.5
            xk1 = xk-tau*grad(xk)
        
        xk = xk1
        res = np.append(res,nla.norm(grad(xk)))
        iterations+=1
    x_sol = xk1
    return x_sol,res

# Condition No. k = 1


if __name__ == "__main__":
        
    [D, c] = create_classification_problem(200, 20, 1)
    x0 = np.zeros((np.size(D,1),1))
    x_sol,res = grad_descent(lambda x:logreg_objective(x,D,c), lambda x:logreg_grad(x,D,c),x0)
    plt.semilogy(res)
    plt.xlabel('No of Iterations')
    plt.ylabel('Residual (Base 10 log scale)')
    plt.title('Condition # (k) = 1')
              
    


