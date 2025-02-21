import matplotlib.pyplot as plt
import numpy as np
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve

# Processes from Lectures
def newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1; nJ=0; #function and Jacobian evals
    npn=1;

    if (len(x0)<100):
        if (np.linalg.cond(Jf(x0)) > 1e16):
            print("Error: matrix too close to singular");
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
            r=x0;
            return (r,rn,nf,nJ);

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        # compute n x n Jacobian matrix
        Jn = Jf(xn);
        nJ+=1;

        if verb:
            print("|--%d--|%1.7f|%1.15f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

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
        if np.linalg.norm(Fn)>Tol:
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

def iteration_method(x0,F,J,nMax,Tol, verb = False):
    n = 0
    err = 1
    xn  = x0
    Fx = F(xn)
    if verb:
            print("|--%d--|%1.7f|%1.15f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fx)));
    while n<nMax and err > Tol:
        Fx = F(xn)
        xn1 = xn - J@Fx
        n+=1
        err = np.linalg.norm(xn1-xn)
        xn = xn1
        if verb:
            print("|--%d--|%1.7f|%1.15f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fx)));
    if verb:
        if np.linalg.norm(xn)>Tol:
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nMax,np.linalg.norm(Fx)));
        else:
            print("Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fx)));
    
    return xn,n


# Problem 1 ==============================================================
def F(x):
        return np.array([3*x[0]**2-x[1]**2 , 3*x[0]*x[1]**2-x[0]**3-1 ]);
J= np.array([[1/6 , 1/18],[0,1/6]]);

x0 = np.array([1,1]); Tol=1e-14; nMax=200;
xn,n = iteration_method(x0,F,J,nMax,Tol, verb = True);
print(xn)

def Jf(x):
    return np.array([[6*x[0] , -2*x[1]],[3*x[1]**2-3*x[1]**2,6*x[0]*x[1]]])

r,rn,nf,nJ = newton_method_nd(F,Jf,x0,Tol,nMax,verb=False)


# Problem 3 ==============================================================
def F(x):
    return np.array([x[0]**2 + 4*x[1]**2 + 4*x[2]**2-16])
    
def gradF(x):
    return np.array([2*x[0],8*x[1],8*x[2]])
    
def gradient_ascent(x0,F,gradF,nMax,tol,verb = False):
    n = 0
    xseq = []
    fseq = []
    xseq.append(x0)
    fseq.append(F(x0))
    if verb:
        print("|--n--|----xn----|---|f(xn)|---|")
        print("|--%d--|%1.7f|%1.15f|" %(n,np.linalg.norm(x0),np.linalg.norm(F(x0))));
    while np.abs(F(x0))>tol and n<nMax:
        d = F(x0)/np.linalg.norm(gradF(x0))**2
        x0 = x0 - d*gradF(x0)
        n+=1
        xseq.append(x0)
        fseq.append(F(x0))
        if verb:
            print("|--%d--|%1.7f|%1.15f|" %(n,np.linalg.norm(x0),np.linalg.norm(F(x0))));
    return x0,n,xseq,fseq

x0 = np.array([1,1,1])
nMax = 100
tol = 1e-10

x0,n,xseq,fseq = gradient_ascent(x0,F,gradF,nMax,tol,True)
print(x0)
print(n)

plt.plot(np.log10(np.abs(fseq)))

plt.grid()
plt.xlabel("Iterations")
plt.ylabel("log10(|F(xn)|)")
plt.title("Log of error per iteration")
