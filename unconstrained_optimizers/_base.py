import numpy as np
import math

# Utilities of cost function
def logistic(x):
    return np.log(1+np.exp(x))

def cost_logistic(A,y,x,lbda):
    yAx = y * A.dot(x)
    return np.mean(logistic(-yAx)) + lbda * np.linalg.norm(x) ** 2 / 2.

def cost_logistic_lasso(A,y,x,lbda):
    yAx = y * A.dot(x)
    return np.mean(logistic(-yAx)) + lbda * np.linalg.norm(x,1)

# Gradient Computations
def compute_grad(A,y,x,lbda):
    yAx = y * A.dot(x)
    n = A.shape[0]
    aux = 1. / (1. + np.exp(yAx))
    return - (A.T).dot(y * aux) / n + lbda * x

def grad_i(A, i, x, y, lbda):
    grad = - A[i] * y[i] / (1. + np.exp(y[i]* A[i].dot(x)))
    grad += lbda * x
    return grad   

# GD implementation     
def gradient_descent(A,y,tau,niter,lbda,A1,y1,change,inference=False):
    d = A.shape[1]
    x_init = np.random.randn(d)
    x = x_init
    cost_values, val_cost_values = [], []
    grad_values = []
    for i in range(niter):
        gd = compute_grad(A,y,x,lbda)
        grad_values.append(np.linalg.norm(gd))
        x = x-tau*gd
        func = cost_logistic(A,y,x,lbda)
        cost_values.append(func)
        if inference:
          val_func = cost_logistic(A1,y1,x,lbda+change)
          val_cost_values.append(val_func)
    if inference:
      return x,cost_values,val_cost_values
    else:
      return x,cost_values,grad_values

# SGD implementation
def stochastic_gradient(A,tau,y,lbda,nb,niter) :
    d = A.shape[1]
    n = A.shape[0]
    x_init = np.random.randn(d)
    x = x_init
    cost_values = []
    val_cost_values = []
    for k in range(niter): 
        ik = np.random.choice(n,nb,replace=False)
        sg = np.zeros(d)
        for j in range(nb):
            gi = grad_i(A,ik[j],x,y,lbda)
            sg = sg + gi
        sg = (1/nb)*sg
        x = x - tau * sg
        if ((k*nb) % n) == 0: 
              func = cost_logistic(A,y,x,lbda)
              cost_values.append(func)   
    return x, cost_values

def stochastic_gradient_(A,tau,y,lbda,nb, niter) :
    d = A.shape[1]
    n = A.shape[0]
    x_init = np.random.randn(d)
    x = x_init
    cost_values = []
    for k in range(niter): 
        ik = np.random.choice(n,nb,replace=False)
        sg = np.zeros(d)
        for j in range(nb):
            gi = grad_i(A,ik[j],x,y,lbda)
            sg = sg + gi
        sg = (1/nb)*sg
        x = x - tau * sg
        func = cost_logistic(A,y,x,lbda)
        cost_values.append(func)
    return x, cost_values

# Proximal Gradient implementation
def Soft(x,s): return np.maximum( abs(x)-s, np.zeros(x.shape)  ) * np.sign(x)

def ISTA(A,y,x,lbda,tau): 
    gd = compute_grad(A,y,x,lbda)
    return Soft( x-tau*(gd ), lbda*tau )

def proximal_gradient(A,y,lbda,niter,tau,d):
    E_test = []
    flist = np.zeros((niter,1))
    x = np.zeros(d)
    for i in np.arange(0,niter):
        flist[i] = cost_logistic_lasso(A,y,x,lbda)
        x = ISTA(A,y,x,lbda,tau)
        yAx = y * A.dot(x)
        E_test.append(np.mean(logistic(-yAx)))
    return(flist,x,E_test)

# SVRG implementation
def svrg(A,y,lbda,n_iter=100,m=5): 
    objvals = []
    n = A.shape[0]
    L = np.linalg.norm(A, ord=2) ** 2 / (4. * n) 
    alpha = 0.2/L
    w0 = np.random.randn(A.shape[1])
    w = w0.copy()
    k=0
    obj = cost_logistic(A,y,w,lbda)
    objvals.append(obj);
    while (k < n_iter):
        gwk = compute_grad(A,y,w,lbda)
        if (k+n)//n > k//n:
            objvals.append(obj)
        wtilda = w
        wtildavg = w
        for j in range(m):
            ij = np.random.choice(n,1,replace=True)
            sg = grad_i(A, ij[0], wtilda, y, lbda)-grad_i(A, ij[0], w, y, lbda)+gwk
            wtilda[:] = wtilda - alpha*sg 
            if (k+n+j)//n > (k+n)//n:
                objvals.append(obj)
        w[:] = wtilda.copy()
        obj = cost_logistic(A,y,w,lbda)
        k += 1
        if k+m+n % n == 0:
            objvals.append(obj)
    if k+m+n % n > 0:
        objvals.append(obj)
    w_output = w.copy()
    return w_output, np.array(objvals)

# Heavy Ball implementation
def heavy_ball(A,y,tau,niter,lbda,gamma):
    d = A.shape[1]
    x_init = np.random.randn(d)
    x = x_init
    x_history = [x]
    cost_values = [cost_logistic(A,y,x,lbda)]
    m = compute_grad(A,y,x,lbda) 
    # Gamma coefficient is introduced mainy but can be computed using the following formula :
    #gamma = (math.sqrt(L)-math.sqrt(mu))**2 / (math.sqrt(L)+math.sqrt(mu))**2 
    #step = 1 / math.sqrt(mu*L)
    for i in range(1,niter):
        m = (1-gamma)*compute_grad(A,y,x,lbda) + gamma*m 
        x = x - tau*m 
        x_history.append(x)
        func = cost_logistic(A,y,x,lbda)
        cost_values.append(func)
    return x,cost_values,gamma

# Non convex case with GD

def cost_logistic_non_convex(A,y,x,lbda,p): # add non convex penalty
    yAx = y * A.dot(x)
    return np.mean(logistic(-yAx)) + lbda * np.sum(np.abs(x)**p)**(1/p)

def compute_grad_non_convex(A,y,x,lbda,p):
    yAx = y * A.dot(x)
    n = A.shape[0]
    aux = 1. / (1. + np.exp(yAx))
    return - (A.T).dot(y * aux) / n + lbda * np.sum(np.abs(x)**(p-1))

# GD implementation     
def gradient_descent_non_convex(A,y,tau,niter,lbda,p,A1,y1,change,inference=False):
    d = A.shape[1]
    x_init = np.random.randn(d)
    x = x_init
    cost_values, val_cost_values = [], []
    grad_values = []
    for i in range(niter):
        gd = compute_grad_non_convex(A,y,x,lbda,p)
        grad_values.append(np.linalg.norm(gd))
        x = x-tau*gd
        func = cost_logistic_non_convex(A,y,x,lbda,p)
        cost_values.append(func)
        if inference:
          val_func = cost_logistic_non_convex(A1,y1,x,lbda+change,p)
          val_cost_values.append(val_func)
    if inference:
      return x,cost_values,val_cost_values
    else:
      return x,cost_values,grad_values
