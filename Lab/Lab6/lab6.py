import numpy as np
from numpy import random as rand
import time
import math
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, Video
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer


# Pre-written processes
# Newton method in n dimensions implementation
def newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1; nJ=0; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        # compute n x n Jacobian matrix
        Jn = Jf(xn);
        nJ+=1;

        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

# Lazy Newton method (chord iteration) without LU (solve is still n^3)
def lazy_newton_method_nd_simple(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1; nJ=0; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        # compute n x n Jacobian matrix only if n==0
        if (n==0):
            Jn = Jf(xn);
            nJ+=1;

        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step
        pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Lazy Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Lazy Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

# Lazy Newton method (chord iteration) in n dimensions implementation
def lazy_newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    # compute n x n Jacobian matrix (ONLY ONCE)
    Jn = Jf(xn);

    # Use pivoted LU factorization to solve systems for Jf. Makes lusolve O(n^2)
    lu, piv = lu_factor(Jn);

    n=0;
    nf=1; nJ=1; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:

        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -lu_solve((lu, piv), Fn); #We use lu solve instead of pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Lazy Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Lazy Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);


def slacker_newton1(f,Jf,x0,updates,tol,nmax,verb=False):
    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    # compute n x n Jacobian matrix (ONLY ONCE)
    n=0;
    

    # Use pivoted LU factorization to solve systems for Jf. Makes lusolve O(n^2)
    

    
    nf=1; nJ=1; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        if n%updates ==0:
            Jn = Jf(xn);
            #print("Jacobian is updated")
        lu, piv = lu_factor(Jn);
            
        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -lu_solve((lu, piv), Fn); #We use lu solve instead of pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Slacker Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Slacker Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

def slacker_newton2(f,Jf,x0,tol,nmax,verb=False):
    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    # compute n x n Jacobian matrix (ONLY ONCE)
    n=0;
    

    # Use pivoted LU factorization to solve systems for Jf. Makes lusolve O(n^2)
    nf=1; nJ=1; #function and Jacobian evals
    npn=1;
    step = npn

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");
    Jn = Jf(xn);
    while npn>tol and n<=nmax:
        if npn*10**-n < tol:
            Jn = Jf(xn);
            #print("Jacobian is updated")
        lu, piv = lu_factor(Jn);
            
        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -lu_solve((lu, piv), Fn); #We use lu solve instead of pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;
        

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Slacker Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Slacker Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);


def fd(f,s,h):
    return (f(s+h)-f(h))/h
    
############################################################################
############################################################################
# Rootfinding example start. You are given F(x)=0.
#First, we define F(x) and its Jacobian.
def F(x):
        return np.array([4*x[0]**2 + x[1]**2 -4,x[0]+x[1]-np.sin(x[0]-x[1])]);
def JF(x):
        return np.array([[8*x[0],2*x[1]],[1-np.cos(x[0]-x[1]),1+np.cos(x[0]-x[1])]]);

    # Apply Newton Method:
x0 = np.array([1,0]); tol=1e-10; nmax=100;
(rN,rnN,nfN,nJN) = newton_method_nd(F,JF,x0,tol,nmax,False);
print(rN)

    # Apply Lazy Newton (chord iteration)
nmax=1000;
(rLN,rnLN,nfLN,nJLN) = lazy_newton_method_nd(F,JF,x0,tol,nmax,False);

    # Plots and comparisons
#numN = rnN.shape[0];
#errN = np.max(np.abs(rnN[0:(numN-1)]-rN),1);
#plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
#plt.title('Newton iteration log10|r-rn|');
#plt.legend();
#plt.show();

#numLN = rnLN.shape[0];
#errLN = np.max(np.abs(rnLN[0:(numLN-1)]-rN),1);
#plt.plot(np.arange(numN-1),np.log10(errN+1e-18),'b-o',label='Newton');
#plt.plot(np.arange(numLN-1),np.log10(errLN+1e-18),'r-o',label='Lazy Newton');
#plt.title('Newton and Lazy Newton iterations log10|r-rn|');
#plt.legend();
#plt.show();

updates = 3
r,rn,nf,nJ = slacker_newton1(F,JF,x0,updates,tol,nmax,True)
#r,rn,nf,nJ = slacker_newton2(F,JF,x0,tol,nmax,True)

################################################################################

# Calculating the jacobian using the forward difference
def F(x):
        return np.array([4*x[0]**2 + x[1]**2 -4,x[0]+x[1]-np.sin(x[0]-x[1])]);
def f1(x):
    return 4*x[0]**2 + x[1]**2 -4
def f2(x):
        return x[0]+x[1]-np.sin(x[0]-x[1])
h = 1e-4
def Jf_approx(x):
    return np.array([[fd(f1,x[0],h),fd(f1,x[1],h)],[fd(f2,x[0],h),fd(f2,x[1],h)]]);

(rN,rnN,nfN,nJN) = newton_method_nd(F,Jf_approx,x0,tol,nmax,False);
print(rN)
