# get lgwts routine and numpy
from gauss_legendre import *
from gauss_legendre import lgwt

# adaptive quad subroutines
# the following three can be passed
# as the method parameter to the main adaptive_quad() function

def eval_composite_trap(n,a,b,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    
    I_trap = h*f(xnode[0])*1/2
    
    for j in range(1,n):
         I_trap = I_trap+h*f(xnode[j])
    I_trap= I_trap + 1/2*h*f(xnode[n])
    
    return I_trap ,xnode, None

def eval_composite_simpsons(n,a,b,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    I_simp = f(xnode[0])

    nhalf = n/2
    for j in range(1,int(nhalf)+1):
         # even part 
         I_simp = I_simp+2*f(xnode[2*j])
         # odd part
         I_simp = I_simp +4*f(xnode[2*j-1])
    I_simp= I_simp + f(xnode[n])
    
    I_simp = h/3*I_simp
    
    return I_simp,xnode,None  

def eval_gauss_quad(M,a,b,f):
  
  #Non-adaptive numerical integrator for \int_a^b f(x)w(x)dx
  # Input:
  #   M - number of quadrature nodes
  #   a,b - interval [a,b]
  #   f - function to integrate
  
  # Output:
  #   I_hat - approx integral
  #   x - quadrature nodes
  #   w - quadrature weights

  # Currently uses Gauss-Legendre rule
  # """
  x,w = lgwt(M,a,b)
  I_hat = np.sum(f(x)*w)
  return I_hat,x,w

def adaptive_quad(a,b,f,tol,M,method):
  # """
  # Adaptive numerical integrator for \int_a^b f(x)dx
  
  # Input:
  # a,b - interval [a,b]
  # f - function to integrate
  # tol - absolute accuracy goal
  # M - number of quadrature nodes per bisected interval
  # method - function handle for integrating on subinterval
  #        - eg) eval_gauss_quad, eval_composite_simpsons etc.
  
  # Output: I - the approximate integral
  #         X - final adapted grid nodes
  #         nsplit - number of interval splits
  # """
  # 1/2^50 ~ 1e-15
  maxit = 50
  left_p = np.zeros((maxit,))
  right_p = np.zeros((maxit,))
  s = np.zeros((maxit,1))
  left_p[0] = a; right_p[0] = b;
  # initial approx and grid
  s[0],x,_ = method(M,a,b,f);
  # save grid
  X = []
  X.append(x)
  j = 1;
  I = 0;
  nsplit = 1;
  while j < maxit:
    # get midpoint to split interval into left and right
    c = 0.5*(left_p[j-1]+right_p[j-1]);
    # compute integral on left and right spilt intervals
    s1,x,_ = method(M,left_p[j-1],c,f); X.append(x)
    s2,x,_ = method(M,c,right_p[j-1],f); X.append(x)
    if np.max(np.abs(s1+s2-s[j-1])) > tol:
      left_p[j] = left_p[j-1]
      right_p[j] = 0.5*(left_p[j-1]+right_p[j-1])
      s[j] = s1
      left_p[j-1] = 0.5*(left_p[j-1]+right_p[j-1])
      s[j-1] = s2
      j = j+1
      nsplit = nsplit+1
    else:
      I = I+s1+s2
      j = j-1
      if j == 0:
        j = maxit
  return I,np.unique(X),nsplit

