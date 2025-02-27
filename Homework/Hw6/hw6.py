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

## DEFINE ALL SUBROUTINES ###############################################
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

# Implementation of Broyden method. B0 can either be an approx of Jf(x0) (Bmat='fwd'),
# an approx of its inverse (Bmat='inv') or the identity (Bmat='Id')
def broyden_method_nd(f,B0,x0,tol,nmax,Bmat='Id',verb=False):

    # Initialize arrays and function value
    d = x0.shape[0];
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1;
    npn=1;

    #####################################################################
    # Create functions to apply B0 or its inverse
    if Bmat=='fwd':
        #B0 is an approximation of Jf(x0)
        # Use pivoted LU factorization to solve systems for B0. Makes lusolve O(n^2)
        lu, piv = lu_factor(B0);
        luT, pivT = lu_factor(B0.T);

        def Bapp(x): return lu_solve((lu, piv), x); #np.linalg.solve(B0,x);
        def BTapp(x): return lu_solve((luT, pivT), x) #np.linalg.solve(B0.T,x);
    elif Bmat=='inv':
        #B0 is an approximation of the inverse of Jf(x0)
        def Bapp(x): return B0 @ x;
        def BTapp(x): return B0.T @ x;
    else:
        Bmat='Id';
        #default is the identity
        def Bapp(x): return x;
        def BTapp(x): return x;
    ####################################################################
    # Define function that applies Bapp(x)+Un*Vn.T*x depending on inputs
    def Inapp(Bapp,Bmat,Un,Vn,x):
        rk=Un.shape[0];

        if Bmat=='Id':
            y=x;
        else:
            y=Bapp(x);

        if rk>0:
            y=y+Un.T@(Vn@x);

        return y;
    #####################################################################

    # Initialize low rank matrices Un and Vn
    Un = np.zeros((0,d)); Vn=Un;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        if verb:
            print("|--%d--|%1.7f|%1.12f|" % (n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        #Broyden step xn = xn -B_n\Fn
        dn = -Inapp(Bapp,Bmat,Un,Vn,Fn);
        # Update xn
        xn = xn + dn;
        npn=np.linalg.norm(dn);

        ###########################################################
        ###########################################################
        # Update In using only the previous I_n-1
        #(this is equivalent to the explicit update formula)
        Fn1 = f(xn);
        dFn = Fn1-Fn;
        nf+=1;
        I0rn = Inapp(Bapp,Bmat,Un,Vn,dFn); #In^{-1}*(Fn+1 - Fn)
        un = dn - I0rn;                    #un = dn - In^{-1}*dFn
        cn = dn.T @ (I0rn);                # We divide un by dn^T In^{-1}*dFn
        # The end goal is to add the rank 1 u*v' update as the next columns of
        # Vn and Un, as is done in, say, the eigendecomposition
        Vn = np.vstack((Vn,Inapp(BTapp,Bmat,Vn,Un,dn)));
        Un = np.vstack((Un,(1/cn)*un));

        n+=1;
        Fn=Fn1;
        rn = np.vstack((rn,xn));

    r=xn;

    if verb:
        if npn>tol:
            print("Broyden method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Broyden method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return(r,rn,nf)

def LS_Gw(f,xn,Fn,dn,nf,eps,maxbis,verb,LS):
    #Derivative-free linesearch for rootfinding
    #Newton and Quasi-Newton methods (Griewank LS method)

    # Begin line search. Evaluate Fn at full step
    Fnp = f(xn+dn);
    nf+=1;
    beta=1;
    ndn = np.linalg.norm(dn);

    if (LS and ndn > 1e-10):
        dFn = Fnp-Fn; #difference in function evals
        nrmd2 = dFn.T @ dFn; #|Fn|^2 = <Fn,Fn>
        q = -(Fn.T @ dFn)/nrmd2; #quality measure q

        #if verb:
        #    print("q0=%1.1e, beta0 = %1.1e" %(q,beta));

        bis=0;
        while q<0.5+eps and bis<maxbis:
            beta=0.5*beta; #halve beta and try again
            Fnp = f(xn+beta*dn);
            dFn = Fnp-Fn;
            nf+=1;
            nrmd2 = dFn.T @ dFn; #|Fn|^2 = <Fn,Fn>
            q = -(Fn.T @ dFn)/nrmd2; #quality measure q
            bis+=1; #increase bisection counter

    pm = beta*dn;
    nrmpn = beta*ndn;
    xn = xn+beta*dn;
    Fn = Fnp;

    return (xn,Fn,nrmpn,nf,beta);

def broyden_method_ndLS(f,B0,x0,tol,nmax,Bmat='Id',verb=False,LS=True):

    # Initialize arrays and function value
    d = x0.shape[0];
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    nrmpn = 1;
    n=0;
    nf=1;

    #linesearch parameters
    maxbis=6; eps=1e-5;

    #####################################################################
    # Create functions to apply B0 or its inverse
    if Bmat=='fwd':
        #B0 is an approximation of Jf(x0)
        # Use pivoted LU factorization to solve systems for B0. Makes lusolve O(n^2)
        lu, piv = lu_factor(B0);
        luT, pivT = lu_factor(B0.T);

        def Bapp(x): return lu_solve((lu, piv), x); #np.linalg.solve(B0,x);
        def BTapp(x): return lu_solve((luT, pivT), x) #np.linalg.solve(B0.T,x);
    elif Bmat=='inv':
        #B0 is an approximation of the inverse of Jf(x0)
        def Bapp(x): return B0 @ x;
        def BTapp(x): return B0.T @ x;
    else:
        Bmat='Id';
        #default is the identity
        def Bapp(x): return x;
        def BTapp(x): return x;
    ####################################################################
    # Define function that applies Bapp(x)+Un*Vn.T*x depending on inputs
    def Inapp(Bapp,Bmat,Un,Vn,x):
        rk=Un.shape[0];

        if Bmat=='Id':
            y=x;
        else:
            y=Bapp(x);

        if rk>0:
            y=y+Un.T@(Vn@x);

        return y;
    #####################################################################

    # Initialize low rank matrices Un and Vn
    Un = np.zeros((0,d)); Vn=Un;
    beta=1; type='broyden';

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|---beta---|--------|---nfv---|");

    while nrmpn>tol and n<=nmax:
        if verb:
            print("|--%d--|%1.7f|%1.12f|%1.3f|%s|%d" % (n,np.linalg.norm(xn),np.linalg.norm(Fn),beta,type,nf));

        #Broyden step xn = xn -B_n\Fn
        if (n==0):
            dn = -Inapp(Bapp,Bmat,Un,Vn,Fn);
        elif (n==1):
            dn = -IFnp - Un.T@(Vn@Fn);
        else:
            dn = -IFnp - (Vn[n-1]@Fn)*Un[n-1];
            #dn = -Inapp(Bapp,Bmat,Un,Vn,Fn);

        ########################################################
        # Derivative-free line search. If full step is accepted (beta=1), this is
        # equivalent to updating xn = xn + dn, Fn = fun(Fn), nrmpn = norm(pn)
        (xn,Fn,nrmpn,nf,beta)=LS_Gw(f,xn,Fn,dn,nf,eps,maxbis,verb,LS);
        ###########################################################
        # Update In using only the previous I_n-1
        #(this is equivalent to the explicit update formula)
        IFnp = Inapp(Bapp,Bmat,Un,Vn,Fn);
        un = (1-beta)*dn + IFnp;
        cn = beta*dn.T @ (dn+IFnp);
        # The end goal is to add the rank 1 u*v' update as the next columns of
        # Vn and Un, as is done in, say, the eigendecomposition
        Vn = np.vstack((Vn,Inapp(BTapp,Bmat,Vn,Un,beta*dn)));
        Un = np.vstack((Un,-(1/cn)*un));

        n+=1;
        rn = np.vstack((rn,xn));

    r=xn;

    if verb:
        if nrmpn>tol:
            print("Broyden method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Broyden method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return(r,rn,nf)

def newton_method_nd_LS(f,Jf,x0,tol,nmax,verb=False,LS=True):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1; nJ=0; #function and Jacobian evals
    npn=1;

    #linesearch parameters
    maxbis=8; eps=1e-1; beta=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|--beta--|");

    while npn>tol and n<=nmax:
        # compute n x n Jacobian matrix
        Jn = Jf(xn);
        nJ+=1;

        if verb:
            print("|--%d--|%1.7f|%1.12f|%1.3f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn),beta));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -np.linalg.solve(Jn,Fn);

        ########################################################
        # Derivative-free line search. If full step is accepted (beta=1), this is
        # equivalent to updating xn = xn + dn, Fn = fun(Fn), nrmpn = norm(pn)
        (xn,Fn,npn,nf,beta)=LS_Gw(f,xn,Fn,pn,nf,eps,maxbis,verb,LS);
        ###########################################################

        n+=1;
        rn = np.vstack((rn,xn));

    r=xn;

    if verb:
        if npn>tol:
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);


################################################################################
# Backtracking line-search algorithm (to find an for the step xn + an*pn)
def line_search(f,Gf,x0,p,type,mxbck,c1,c2):
    alpha=2;
    n=0;
    cond=False; #condition (if True, we accept alpha)
    f0 = f(x0); # initial function value
    Gdotp = p.T @ Gf(x0); #initial directional derivative
    nf=1;ng=1; # number of function and grad evaluations

    # we backtrack until our conditions are met or we've halved alpha too much
    while n<=mxbck and (not cond):
        alpha=0.5*alpha;
        x1 = x0+alpha*p;
        # Armijo condition of sufficient descent. We draw a line and only accept
        # a step if our function value is under this line.
        Armijo = f(x1) <= f0 + c1*alpha*Gdotp;
        nf+=1;
        if type=='wolfe':
            #Wolfe (Armijo sufficient descent and simple curvature conditions)
            # that is, the slope at new point is lower
            Curvature = p.T @ Gf(x1) >= c2*Gdotp;
            # condition is sufficient descent AND slope reduction
            cond = Armijo and Curvature;
            ng+=1;
        elif type=='swolfe':
            #Symmetric Wolfe (Armijo and symmetric curvature)
            # that is, the slope at new point is lower in absolute value
            Curvature = np.abs(p.T @ Gf(x1)) <= c2*np.abs(Gdotp);
            # condition is sufficient descent AND symmetric slope reduction
            cond = Armijo and Curvature;
            ng+=1;
        else:
            # Default is Armijo only (sufficient descent)
            cond = Armijo;

        n+=1;

    return(x1,alpha,nf,ng);

################################################################################
# Steepest descent algorithm
def steepest_descent(f,Gf,x0,tol,nmax,type='swolfe',verb=True):
    # Set linesearch parameters
    c1=1e-3; c2=0.9; mxbck=10;
    # Initialize alpha, fn and pn
    alpha=1;
    xn = x0; #current iterate
    rn = x0; #list of iterates
    fn = f(xn); nf=1; #function eval
    pn = -Gf(xn); ng=1; #gradient eval

    # if verb is true, prints table of results
    if verb:
        print("|--n--|-alpha-|----|xn|----|---|f(xn)|---|---|Gf(xn)|---|");

    # while the size of the step is > tol and n less than nmax
    n=0;
    while n<=nmax and np.linalg.norm(pn)>tol:
        if verb:
            print("|--%d--|%1.5f|%1.7f|%1.7f|%1.7f|" %(n,alpha,np.linalg.norm(xn),np.abs(fn),np.linalg.norm(pn)));

        # Use line_search to determine a good alpha, and new step xn = xn + alpha*pn
        (xn,alpha,nfl,ngl)=line_search(f,Gf,xn,pn,type,mxbck,c1,c2);

        nf=nf+nfl; ng=ng+ngl; #update function and gradient eval counts
        fn = f(xn); #update function evaluation
        pn = -Gf(xn); # update gradient evaluation
        n+=1;
        rn=np.vstack((rn,xn)); #add xn to list of iterates

    r = xn; # approx root is last iterate

    return (r,rn,nf,ng);

## PROBLEM 1 #######################################################################

# Problem 1
def F(X):
    x = X[0]
    y = X[1]
    return np.array([x**2 + y**2 - 4, np.exp(x)+y-1])

def Jf(x):
    return np.array([[2*x[0], 2*x[1]],[np.exp(x[0]),1]])



    # Apply Newton Method:
x0 = np.array([1,1]); tol=1e-10; nmax=100;
B0 = Jf(x0)

nmax = 250
tol = 1e-10
print("x0 = [1,1]=================================")
try:
    (rNewt1,rnNewt1,nfNewt1,nJNewt1) = newton_method_nd(F,Jf,x0,tol,nmax)
    print(f'Newton: converges to {rNewt1} in {nfNewt1}')
except ValueError:
    print('Newton: no convergence')

try:
    (rLazy1,rnLazy1,nfLazy1,nJLazy1) = lazy_newton_method_nd(F,Jf,x0,tol,nmax)
    print(f'Lazy: converges to {rLazy1} in {nfLazy1}')
except ValueError:
    print('Lazy: no convergence')
    
try:
    (rBroy1,rnBroy1,nfBroy1) = broyden_method_nd(F,B0,x0,tol,nmax,Bmat='Id',verb=False)
    print(f'Broyden: converges to {rBroy1} in {nfBroy1}')
except ValueError:
    print('Broyden: no convergence')
except RuntimeWarning:
    print('Broyden: no convergence')


x0 = np.array([1,-1]); 
B0 = Jf(x0)
print("x0 = [1,-1]=================================")
try:
    (rNewt2,rnNewt2,nfNewt2,nJNewt2) = newton_method_nd(F,Jf,x0,tol,nmax)
    print(f'Newton: converges to {rNewt2} in {nfNewt2}')
except ValueError:
    print('Newton: no convergence')

try:
    (rLazy2,rnLazy2,nfLazy2,nJLazy2) = lazy_newton_method_nd(F,Jf,x0,tol,nmax)
    print(f'Lazy: converges to {rLazy2} in {nfLazy2}')
except ValueError:
    print('Lazy: no convergence')
    
try:
    (rBroy2,rnBroy2,nfBroy2) = broyden_method_nd(F,B0,x0,tol,nmax,Bmat='Id',verb=False)
    print(f'Broyden: converges to {rBroy2} in {nfBroy2}')
except ValueError:
    print('Broyden: no convergence')
except RuntimeWarning:
    print('Broyden: no convergence')


x0 = np.array([0,0]); 
B0 = Jf(x0)
print("x0 = [0,0]=================================")
try:
    (rNewt3,rnNewt3,nfNewt3,nJNewt3) = newton_method_nd(F,Jf,x0,tol,nmax)
    print(f'Newton: converges to {rNewt3} in {nfNewt3}')
except ValueError:
    print('Newton: no convergence')

try:
    (rLazy3,rnLazy3,nfLazy3,nJLazy3) = lazy_newton_method_nd(F,Jf,x0,tol,nmax)
    print(f'Lazy: converges to {rLazy3} in {nfLazy3}')
except ValueError:
    print('Lazy: no convergence')
    
try:
    (rBroy3,rnBroy3,nfBroy3) = broyden_method_nd(F,B0,x0,tol,nmax,Bmat='Id',verb=False)
    print(f'Broyden: converges to {rBroy3} in {nfBroy3}')
except ValueError:
    print('Broyden: no convergence')
except RuntimeWarning:
    print('Broyden: no convergence')

## PROBLEM 2 ########################################################################

# Problem 2
def F(X):
    x = X[0]
    y = X[1]
    z = X[2]
    return np.array([x+np.cos(x*y*z)-1,(1-x)**0.25+y+0.05*z**2-0.15*z-1,-x**2-0.1*y+z-1])

def Jf(X):
    x = X[0]
    y = X[1]
    z = X[2]
    return np.array([[1-y*z*np.sin(x*y*z),-x*y*np.sin(x*y*z),-x*y*np.sin(x*y*z)],[-0.25*(1-x)**-0.75,1,0.1*z-0.15],[-2*x,-0.2*y+0.01,1]])

x0 = np.array([0,0,0])
B0 = Jf(x0)
nmax = 250
tol = 1e-6

print("Newton's Method")
# Newton's Method to Minimize
(rNewt,rnNewt,nfNewt,nJNewt) = newton_method_nd(F,Jf,x0,tol,nmax,True)

# Gradient Descent to Minimize
    #Gradient of q.


    # Define quadratic function and its gradient based on (F,JF)
def q(x):
    Fun = F(x);
    return 0.5*(Fun[0]**2 + Fun[1]**2 + Fun[2]**2);

def Gq(x):
    Jfun = Jf(x);
    Ffun = F(x);
    return np.transpose(Jfun)@Ffun;

print("Steepest Descent:")
    # Apply steepest descent:
(r,rn,nf,ng)=steepest_descent(q,Gq,x0,tol,nmax)
print("Steepest descent converged, n = ", len(rn)," , |F(xn)| = ", np.linalg.norm(F(r)))

print(" ")
print("Hybrid Method")
# Steepest descent then Newton's Method
tol = 5e-2
(r,rn,nf,ng)=steepest_descent(q,Gq,x0,tol,nmax)
(rNewt,rnNewt,nfNewt,nJNewt) = newton_method_nd(F,Jf,r,tol,nmax,True)
print("Hybrid method converged with n = ", (len(rn)+len(rnNewt))," , |F(xn)| = ", np.linalg.norm(F(rNewt)))




