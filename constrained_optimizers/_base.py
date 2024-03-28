import numpy as np

def compute_grad(A,y,x,lbda):
        yAx = y * A.dot(x)
        n = A.shape[0]
        aux = 1. / (1. + np.exp(yAx))
        return - (A.T).dot(y * aux) / n + lbda * x

def logistic(x):
    return np.log(1+np.exp(x))

def cost_logistic(A,y,x,lbda):
        yAx = y * A.dot(x)
        return np.mean(logistic(-yAx)) + lbda * np.linalg.norm(x) ** 2 / 2.



########### CG ###########

#lmo = linear minimization oracle
def lmo_l2(u, l2_constraint):
    u_constraint = np.sqrt(np.dot(u,u))
    x = l2_constraint*(-1*u)/u_constraint
    return x

def lmo_l1(u, l1_constraint):
    num_dim = len(u)
    vertices = l1_constraint*np.concatenate((np.identity(num_dim), -1*np.identity(num_dim)), axis = 0)
    dot_products = np.dot(vertices, u)
    x = vertices[np.argmin(dot_products),:]
    return x

def conditional_gradient(A,y,constraint,niter, method, lbda=0):
    d = A.shape[1]
    x_init = np.random.randn(d)
    x = x_init
    cost_values = []
    for i in range(niter):
        gamma = 2./(i+2)
        gd = compute_grad(A,y,x,lbda)
        if method=="l2":
          s = lmo_l2(gd, constraint)
          x = gamma*s + (1 - gamma)*x
        elif method=="l1":
          s = lmo_l1(gd, constraint)
          x = gamma*s + (1 - gamma)*x
        func = cost_logistic(A,y,x,lbda)
        cost_values.append(func)
    return x,cost_values

def projected_gradient_descent(A,y,tau,niter,projection,constraint,lbda=0):
    d = A.shape[1]
    x_init = np.random.randn(d)
    x = x_init
    cost_values = []
    for i in range(niter):
        gd = compute_grad(A,y,x,lbda)
        x = x - tau*gd
        if projection=="l2":
          x_proj = po_l2(x, constraint)
        elif projection=="l1":
          x_proj = po_l1(x, constraint)
        func = cost_logistic(A,y,x,lbda)
        cost_values.append(func)
    return x,cost_values


########### PGD ###########
def po_l2(w, l2_constraint): 
    norm_w = np.sqrt(np.dot(w,w))
    if norm_w <= l2_constraint:
        w_projected = w
    else:
        w_hat = w/norm_w
        w_projected = w_hat*l2_constraint
    return w_projected

def compute_l1_norm(x):
    x = np.array(x)
    l1_norm = np.sum(np.abs(x))
    return l1_norm

def vector_plus(x): # analogue to relu activation : get postive values and set negative ones to 0
    x = np.array(x)
    mask = x>0
    x_plus = x*mask
    return x_plus

def po_l1(w, l1_constraint):
    w = np.array(w)
    w_sign = 2*(w>0) - 1
    w_pos = np.abs(w)
    w_norm = compute_l1_norm(w)
    if w_norm <= l1_constraint:
        w_projected = w
    else:
        # tolerance value in projection
        epsilon = 10**(-5)
        # original vector is outside l1 ball
        tau_low = 0
        # finding a tau high that pushes given vector into l1 ball
        tau_high = 1
        one_vector = np.ones(len(w))
        forced_vector = vector_plus(w_pos - tau_high*one_vector)
        while compute_l1_norm(forced_vector) > l1_constraint:
            tau_high = 2*tau_high
            forced_vector = vector_plus(w_pos - tau_high*one_vector)
        tau = (tau_high + tau_low)/2
        w_projected = w_sign*vector_plus(w_pos - tau*one_vector) 
        norm_w_projected= compute_l1_norm(w_projected)
        while np.abs(norm_w_projected - l1_constraint) > epsilon:
            # "projection" too far in, so reduce tau
            if norm_w_projected < l1_constraint:
                tau_high = tau
            # "projection" is still too close to the original point
            elif norm_w_projected > l1_constraint:
                tau_low = tau
            tau = (tau_high + tau_low)/2
            w_projected = w_sign*(vector_plus(w_pos - tau*one_vector))
            norm_w_projected = compute_l1_norm(w_projected)
    return w_projected