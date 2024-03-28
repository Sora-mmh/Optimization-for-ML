import numpy as np 
from scipy.linalg import norm 

def logistic(x):
    return np.log(1+np.exp(x))

def cost_logistic(A,y,x,lbda):
        yAx = y * A.dot(x)
        return np.mean(logistic(-yAx)) + lbda * np.linalg.norm(x) ** 2 / 2.

def grad_i(A, i, x, y, lbda):
        grad = - A[i] * y[i] / (1. + np.exp(y[i]* A[i].dot(x)))
        grad += lbda * x
        return grad 

def compute_grad(A,y,x):
        yAx = y * A.dot(x)
        n = A.shape[0]
        aux = 1. / (1. + np.exp(yAx))
        return - (A.T).dot(y * aux) / n 

def randomized_block_coordinate_descent(x0,A,y,lbda,nblocks=1,nits=500): 
    objvals = []
    nnzvals = []
    x = x0.copy()
    k=0    
    n,d = A.shape
    ell = norm(np.matmul(A.T,A),axis=0)  
    obj = cost_logistic(A,y,x,lbda)
    objvals.append(obj)
    nnzvals.append(np.count_nonzero(x))
    g = compute_grad(A,y,x)
    while (k < nits):
        jk = np.random.choice(d,nblocks,replace=False) 
        for j in jk:
            valj = x[j]-(1/ell[j])*g[j]
            threshold = (1/ell[j])*lbda
            if valj < -threshold:
                x[j] = valj+threshold
            elif valj > threshold:
                x[j] = valj-threshold
            else:
                x[j] = 0
        obj = cost_logistic(A,y,x,lbda)
        objvals.append(obj)
        nnzvals.append(np.count_nonzero(x))
        g = compute_grad(A,y,x)
        k += 1  
    x_output = x.copy()   
    return x_output, np.array(objvals), np.array(nnzvals)

def randomized_block_coordinate_descent_with_stochastic_samples(x0,A,y,lbda,nb,nblocks=1,nits=500): 
    objvals = []
    nnzvals = []
    x = x0.copy()
    k=0    
    n,d = A.shape
    ell = norm(np.matmul(A.T,A),axis=0)  
    obj = cost_logistic(A,y,x,lbda)
    objvals.append(obj)
    nnzvals.append(np.count_nonzero(x))
    while (k < nits):
        jk = np.random.choice(d,nblocks,replace=False) 
        ik = np.random.choice(n,nb,replace=False)
        sg = np.zeros(d)
        for j in range(nb):
            gi = grad_i(A,ik[j],x,y,lbda)
            sg = sg + gi
        sg = (1/nb)*sg
        for j in jk:
            valj = x[j]-(1/ell[j])*sg[j]
            threshold = (1/ell[j])*lbda
            if valj < -threshold:
                x[j] = valj+threshold
            elif valj > threshold:
                x[j] = valj-threshold
            else:
                x[j] = 0
        obj = cost_logistic(A,y,x,lbda)
        objvals.append(obj)
        nnzvals.append(np.count_nonzero(x))
        k += 1  
    x_output = x.copy()   
    return x_output, np.array(objvals), np.array(nnzvals)