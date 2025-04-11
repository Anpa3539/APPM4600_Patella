import numpy as np
import matplotlib as plt
import scipy as sc

## PROBLEM 1: FINDING SIMPSON'S APPROXIMATION ########################################
def simpsons_eval(a,b,f):
    I = (b-a)/6*(f(a) + 4*f((a+b)/2) + f(b))
    return I

def composite_simpsons(a,b,nint,f):
    h = (b-a)/nint    
    I = 0
    for i in range(nint):
        I = I + simpsons_eval((a+i*h),(a + (i+1)*h),f)
    return I   
    
f = lambda x: 1/(1+x**2)
a = -5
b = 5
n_simp = 41

I_simp = composite_simpsons(a,b,n_simp,f)
print("Using Simpson's rule: I = ",I_simp)

def trap_eval(a,b,f):
    return (b-a)*(0.5)*(f(a)+f(b))

def composite_trap(a,b,nint,f):
    h = (b-a)/nint    
    I = 0
    for i in range(nint):
        I = I + trap_eval((a+i*h),(a + (i+1)*h),f)
    return I  

n_trap = 410
I_trap = composite_trap(a,b,n_trap,f)
print("Using Trapeziodal rule: I = ",I_trap)


## Compare these numbers to the Scipy quad routine
I_scipy,error, infodict = sc.integrate.quad(f, a, b,full_output=1)
print("scipy integration with tol = 1e-6: I = ",I_scipy," using ", infodict['neval'],' evaluations') 
I_scipy,error, infodict = sc.integrate.quad(f, a, b,epsabs=1e-4,full_output=1)
print("scipy integration with tol = 1e-4: I = ",I_scipy," using ", infodict['neval'],' evaluations') 


## PROBLEM 2: ESTIMATE INDEFINITE INTEGRAL ##########################################################################
def f(x):
    if x == 0:
        return 0
    else:
        return np.cos(1/x)*x
    
a = 0
b = 1
n = 5

I_approx = composite_simpsons(a,b,n,f)
print("The approximation is: ", I_approx)
