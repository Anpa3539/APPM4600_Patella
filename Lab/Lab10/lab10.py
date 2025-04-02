import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
from scipy.integrate import quad

# Pre Lab: Define subroutine for generating Legendre Polynomials

def legendre_poly(x,n):
    p = np.zeros(n+1)
    p[0] = 1
    if n!=0:
        p[1]=x
    for i in range(n):
        p[i+1] = 1/(1+i)*((2*i+1)*x*p[i]-i*p[i-1])
    return p[-1]

def eval_legendre_expansion(f,a,b,w,n,x):
# This subroutine evaluates the Legendre expansion
# Evaluate all the Legendre polynomials at x that are needed
# by calling your code from prelab
    p = lambda j: legendre_poly(x,j)
# initialize the sum to 0
    pval = 0.0
    for j in range(0,n+1):
# make a function handle for evaluating phi_j(x)
        phi_j = lambda x: legendre_poly(x,j)
# make a function handle for evaluating phi_j^2(x)*w(x)
        phi_j_sq = lambda x: legendre_poly(x,j)**2*w(x)
# use the quad function from scipy to evaluate normalizations
        norm_fac,err =  quad(phi_j_sq(x),a,b)
# make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
        func_j = lambda x: phi_j(x)*f(x)*w(x)
# use the quad function from scipy to evaluate coeffs
        num,err = quad(func_j(x),a,b)
        aj,err = x/norm_fac(x)
# accumulate into pval
        pval = pval+aj*p[j]
        return pval


# function you want to approximate
f = lambda x: math.exp(x)
# Interval of interest
a = -1
b = 1
# weight function
w = lambda x: 1.
# order of approximation
n = 2
# Number of points you want to sample in [a,b]
N = 1000
xeval = np.linspace(a,b,N+1)
pval = np.zeros(N+1)
for kk in range(N+1):
    pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])
#’’’ create vector with exact values’’’
fex = np.zeros(N+1)
for kk in range(N+1):
    fex[kk] = f(xeval[kk])
plt.figure()
plt.plot(xeval,fex,'ro-', label= 'f(x)')
plt.plot(xeval,pval,'bs--',label= 'Expansion')
plt.legend()
plt.show()
err = abs(pval-fex)
#plt.semilogy(xeval,err_l,'ro--',label='error')
plt.legend()
plt.show()









