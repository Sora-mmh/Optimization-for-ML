import numpy as np
import random

def dot(x, W):
  x = x[0] if type(x)==tuple else x
  value =  np.dot(W, x)
  def vjp(u):
    return W.T.dot(u.T), np.outer(u, x)
  return value, vjp

def relu(x):
  x = x[0] if type(x)==tuple else x
  value = np.maximum(x, 0)
  gprime = np.zeros(len(x))
  gprime[x >= 0] = 1
  def vjp(u):
    return u * gprime, 
  return value, vjp

def squared_loss(y_pred, y):
  y_pred = y_pred[0]
  diff = y_pred - y
  def vjp(u):
    return diff * u, -diff * u
  return np.array([0.5 * np.sum((y - y_pred) ** 2)]), vjp



def MLP(n, y, seed=0, k=2):
  rng = np.random.RandomState(seed)
  if k==2:
    # default setting for 2 layers
    funcs = [dot, relu, dot, relu, dot, squared_loss]
    params = [rng.randn(3, n), None, rng.randn(4, 3), None, rng.randn(1, 4), y]
  else:
    #create an MLP of k layers
    funcs = []
    min_units, max_units = 3, 7
    hidden_units = {f"layer_{idx}":0 for idx in range(k)}
    for layer_idx in range(k):
        funcs.append(dot)
        funcs.append(relu)
        hidden_units[f"layer_{layer_idx}"] = random.randint(min_units, max_units)
    funcs.append(dot)
    funcs.append(squared_loss)
    units_values = list(hidden_units.values())
    shapes = [(units_values[0], n)]
    for i in range(len(units_values)-1):
      shapes.append((units_values[i+1], units_values[i]))
    shapes.append((1, units_values[-1]))
    params = []
    for i in range(len(shapes)-1):
      params.append(rng.randn(shapes[i][0], shapes[i][1]))
      params.append(None) 
    params.append(rng.randn(shapes[-1][0], shapes[-1][1]))
    params.append(y)
  return funcs, params

# call functions in an appropriate way regarding the parameters
def call_func(x, func, param):
  if param is None:
    return func(x)
  else:
    return func(x, param)

# evaluate the calculus chain
def evaluate_chain(x, funcs, params, return_all=False):
  if len(funcs) != len(params):
    raise ValueError("len(funcs) and len(params) should be equal.")
  L = [x]
  for l in range(len(funcs)):
    L.append(call_func(L[l], funcs[l], params[l]))
  if return_all:
    return L
  else:
    return L[-1]

# call vjps in an appropriate way regarding the parameters of the associated function
def call_vjp(x, func, param, u):
  if param is None:
    _, vjp = func(x)
    vjp_x, = vjp(u)
    vjp_param = None
  else:
    _, vjp = func(x, param)
    vjp_x, vjp_param = vjp(u)
  return vjp_x, vjp_param

# backpropagation
def backward_differntiation_chain(x, funcs, params):
  L = evaluate_chain(x, funcs, params, return_all=True)
  m = L[-1][0].shape[0]  
  K = len(funcs)  
  U = list(np.eye(m))
  J = [None] * K
  for k in reversed(range(K)):
    jac = []
    for i in range(m):
      vjp_x, vjp_param = call_vjp(L[k][0], funcs[k], params[k], U[i])
      jac.append(vjp_param)
      U[i] = vjp_x
    J[k] = np.array(jac)
  J = [j[0] for j in J]
  return L[-1], np.array(U), J

# Train an MLP on SGD
def train(funcs,params,trainset,testset,tau,niter,n,nb):
  cost_values = []
  idxs = [idx for idx,p in enumerate(params) if p is not None]
  params = np.array(params)[idxs]
  funcs = np.array(funcs)[idxs]
  for k in range(niter):
      ik = np.random.choice(n,nb,replace=False)
      for j in range(nb):
          _, jac_wrt_inp, jac_wrt_params =  backward_differntiation_chain(trainset[ik[j]], funcs, params) 
          for idx,param in enumerate(params):
            params[idx] -= tau *jac_wrt_params[idx] 
          cost = evaluate_chain(trainset[ik[j]], funcs, params)[0][0]
          cost_values.append(cost)
  return cost_values

def numerical_jvp(f, x, v, eps=1e-6):
  if not np.array_equal(x.shape, v.shape):
    raise ValueError("x and v should have the same shape.")
  return (f(x + eps * v)[0][0] - f(x - eps * v)[0][0]) / (2 * eps)

# Compute numerical jacobian wrt a parameter
def numerical_jacobian(f, x, eps=1e-6):
  def e(i):
    ret = np.zeros_like(x)
    ret[i] = 1
    return ret
  def E(i, j):
    ret = np.zeros_like(x)
    ret[i, j] = 1
    return ret
  if len(x.shape) == 1:
    return np.array([numerical_jvp(f, x, e(i), eps=eps) for i in range(len(x))]).T
  elif len(x.shape) == 2:
    return np.array([[numerical_jvp(f, x, E(i, j), eps=eps) for i in range(x.shape[0])] for j in range(x.shape[1])]).T
  else:
    raise NotImplementedError