import numpy as np
import matplotlib.pyplot as plt


## PROBLEM 1 #################################################
maclaurin = lambda x: x - x**3/6 + x**5/120
f = lambda x: np.sin(x)
x = np.linspace(0,5,1000)
errmac = np.log10(np.abs(f(x)-maclaurin(x)))

## Problem 1a ################################################

# Solving for a_n,b_n
A = np.array([[1,0,0,0,0,0,0],
              [0,1,0,0,0,0,0],
              [0,0,1,0,-1,0,0],
              [0,0,0,1,0,-1,0],
              [0,0,0,0,-1/6,0,1],
              [0,0,0,0,0,1/6,0],
              [0,0,0,0,1/120,0,-1/6]])
b = np.array([0,1,0,-1/6,0,1/120,0])
a = np.linalg.solve(A,b)

P33 = lambda x: (x-7*x**3/60)/(1+0.05*x**2)
err33 = np.log10(np.abs(P33(x) - f(x)))

## Problem 1b ###############################################
A = np.array([[1,0,0,0,0,0,0],
              [0,1,0,0,0,0,0],
              [0,0,1,-1,0,0,0],
              [0,0,0,0,1,0,0],
              [0,0,0,-1/6,0,1,0],
              [0,0,0,0,1/6,0,-1],
              [0,0,0,1/120,0,-1/6,0]])
b = np.array([0,1,0,1/6,0,1/120,0])
a = np.linalg.solve(A,b)


P42 = lambda x: (x)/(1+x**2/6+7*x**4/360)
err42 = np.log10(np.abs(P42(x) - f(x)))

## Problem 1c ###############################################
P24 = lambda x: (x-7*x**3/60)/(1+x**2/20)
err24 = np.log10(np.abs(P24(x) - f(x)))

## Plotting results for problem 1 ###########################

plt.figure()
plt.plot(x,err33,label="P_3^3(x)")
plt.plot(x,err42,label="P_4^2(x)")
plt.plot(x,err24,label="P_2^4(x)")
plt.plot(x,errmac,label="Maclaurin deg 6")
plt.grid()
plt.xlabel("x")
plt.ylabel("log(error)")
plt.legend()
plt.title("Error for estimating sin(x) using Pade and Maclaurin Approximations")
plt.show()

## Problem 2 #################################################
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

def J(x):
    return np.array([[1,0,0],[x[2],0.5,x[0]],[x[2]**2,x[1],2*x[2]*x[1]]])

def F(x):
    c1 = x[0]
    x0 = x[1]
    x1 = x[2]
    return np.array([[-.5 + c1],[0.5*x0 + c1*x1 - 0.5],[0.5*x0**2+c1*x1**2-1/3]])

x0 = np.array([0.5,0,0])
tol = 1e-9
nmax = 100
r,rn,nf,nJ = newton_method_nd(F,J,x0,tol,nmax,True)

