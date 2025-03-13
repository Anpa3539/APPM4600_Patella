import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv 
from numpy.linalg import norm


# Functions to calculate interpolation nodes
def getx(N,xmin,xmax):
    i = np.zeros(N)
    x = np.zeros(N)
    h = np.abs(xmax-xmin)/(N-1)
    for j in range(N):
        i[j] = j + 1 # Getting it to be 1,2,...
        x[j] = xmin + (i[j]-1)*h
    return x


# def getXchev(N, xmin, xmax):
#     """Generate Chebyshev nodes properly scaled to the given interval [xmin, xmax]."""
#     i = np.arange(N+1)
#     x = (xmin + xmax) / 2 + (xmax - xmin) / 2 * np.cos((2*i + 1) * np.pi / (2 * (N+1)))
#     x = np.sort(x)
#     return x

# def getXchev(N, xmin, xmax):
#     """Generate Chebyshev nodes scaled to [xmin, xmax], ensuring exact endpoints in increasing order."""
#     i = np.arange(1, N)  # Exclude first and last nodes for now
#     x_cheb = np.cos((2*i + 1) * np.pi / (2 * (N+1)))  # Standard Chebyshev formula

#     # Scale nodes to [xmin, xmax]
#     x_cheb = (xmin + xmax) / 2 + (xmax - xmin) / 2 * x_cheb

#     # Manually set the exact endpoints and ensure increasing order
#     x = np.concatenate(([xmin], np.sort(x_cheb), [xmax]))
#     x = np.sort(x)

#     return x

def getXchev(n, a=-1, b=1):
  
  if n <= 0:
    return np.array([])
  indices = np.arange(1, n + 1)
  nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * indices - 1) * np.pi / (2 * n))
  nodes = np.sort(nodes)
  return nodes


# def getXchev(N,xmin,xmax):
#     xint = []
#     for j in range(1, N+1):
#         x_j = np.cos((((2*j)-1)*np.pi)/(2*N))
#         xint.append(xmax*x_j)
#     xint.append(0)
#     xint = np.array(xint)
#     xint = np.sort(xint)
#     return(xint)


## PROBLEM 1: INTERPOLATE FUNCTION USING EQUISPACED NODES
f = lambda x: 1 + 1/(1+x**2)
fp = lambda x: -2*x/(1.+x**2)**2
n = np.array([5,10,15,20])
xmin = -5
xmax = 5
xint = np.linspace(xmin,xmax,1001)
xeval = np.linspace(xmin,xmax,1001)
feval = f(xeval)
yint = f(xint)
ypint = fp(xint)

def eval_lagrange(xeval,xint,yint,N):
    lj = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])
            
    yeval = 0.
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
    return(yeval)

def eval_hermite(xeval,xint,yint,ypint,N):

#     ''' Evaluate all Lagrange polynomials'''

    lj = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N+1)
#    lpj2 = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
#              lpj2[count] = lpj2[count]*(xint[count] - xint[jj])
              lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])
    yeval = 0.
    for jj in range(N+1):
       Qj = (1.-2.*(xeval-xint[jj])*lpj[jj])*lj[jj]**2
       Rj = (xeval-xint[jj])*lj[jj]**2
       yeval = yeval + yint[jj]*Qj+ypint[jj]*Rj 
    return(yeval)

def create_natural_spline(yint,xint,N):
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)  
    for i in range(1,N):
       hi = xint[i]-xint[i-1]
       hip = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
       h[i-1] = hi
       h[i] = hip
#  create matrix so you can solve for the M values
# This is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    A[0][0] = 1.0
    for j in range(1,N):
       A[j][j-1] = h[j-1]/6
       A[j][j] = (h[j]+h[j-1])/3 
       A[j][j+1] = h[j]/6
    A[N][N] = 1
    Ainv = inv(A)
    M  = Ainv.dot(b)
#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = yint[j]/h[j]-h[j]*M[j]/6
       D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
    return(M,C,D)
       
def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):

    hi = xip-xi
    yeval = (Mi*(xip-xeval)**3 +(xeval-xi)**3*Mip)/(6*hi) \
            + C*(xip-xeval) + D*(xeval-xi)
    return yeval 
    
def  eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        
#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
        yeval[ind] = yloc

    return(yeval)

def create_clamped_spline(yint, xint, N, fp0, fpN):
    b = np.zeros(N+1)
    h = np.zeros(N+1)

    for i in range(1, N):
        hi = xint[i] - xint[i-1]
        hip = xint[i+1] - xint[i]
        b[i] = (yint[i+1] - yint[i]) / hip - (yint[i] - yint[i-1]) / hi
        h[i-1] = hi
        h[i] = hip

    # Create matrix A
    A = np.zeros((N+1, N+1))
    
    # First row (clamped condition at x0)
    A[0][0] = 2 / h[0]
    A[0][1] = 1 / h[0]
    b[0] = 6 * ((yint[1] - yint[0]) / h[0] - fp0) / h[0]

    for j in range(1, N):
        A[j][j-1] = h[j-1] / 6
        A[j][j] = (h[j] + h[j-1]) / 3
        A[j][j+1] = h[j] / 6

    # Last row (clamped condition at xN)
    A[N][N-1] = 1 / h[N-1]
    A[N][N] = 2 / h[N-1]
    b[N] = 6 * (fpN - (yint[N] - yint[N-1]) / h[N-1]) / h[N-1]

    # Solve for M values
    M = np.linalg.solve(A, b)

    # Compute coefficients C and D
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
        C[j] = yint[j] / h[j] - h[j] * M[j] / 6
        D[j] = yint[j+1] / h[j] - h[j] * M[j+1] / 6

    return M, C, D
def getPhi(xi,x): # CHECKED
    phi = 1
    for i in range(len(xi)):
        phi = phi * (x - xi[i])
    return phi

# Subroutine to call in main function to get the list of omega values
def getOmega(x): # CHECKED
    omega = np.ones(len(x))
    for j in range(len(x)):
        
        for i in range(len(x)):
            if x[i] != x[j]:
                omega[j] = omega[j]*(x[j]-x[i])**(-1)
    return omega
def barycentric_lagrange(x,y,xq):
    p = np.zeros(len(xq))
    omega = getOmega(x)
    
    for i in range(len(xq)):
        phi = getPhi(x,xq[i])
        sum1 = 0
        for j in range(len(x)):
            if xq[i] == x[j]:
                sum1 = y[j]
            else:
                sum1 = sum1 + omega[j]/(xq[i] - x[j])*y[j]

           
        p[i] = phi*sum1

    return(p)
# Solve using lagrange interpolation =========================================================
eval_lag = []
xeval = np.linspace(xmin,xmax,1001)
for i in range(len(n)):

    # Getting x
    xint = getXchev(n[i]+1,xmin,xmax)
    yint = f(xint)
    eval_lagi = np.zeros(len(xeval))

    # Calculating the value of the interpolation
    for k in range(len(xeval)):
        eval_lagi[k] = eval_lagrange(xeval[k],xint,yint,n[i])
    eval_lag.append(eval_lagi)
    
    

    # Plotting results
    plt.figure()
    plt.plot(xeval,eval_lagi,label = "p(x)")
    plt.scatter(xint,yint,label = "Interplation nodes")
    plt.plot(xeval,feval,label = "f(x)")
    plt.title(f"N = {n[i]}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()


# Creating the error plot
err = []
plt.figure()
for j in range(len(n)):
    err.append(np.log10(np.abs(eval_lag[j] - feval)))
    plt.plot(xeval,err[j],label = f"n = {n[j]}")

plt.grid()
plt.xlabel("x")
plt.ylabel("log(error)")
plt.title("Interpolation error for different values of n: Lagrange")
plt.legend()




## SOLVING USING Hermite  #####################################################################
eval_her = []
for i in range(len(n)):

    # Getting x
    xint = getXchev(n[i]+1,xmin,xmax)
    yint = f(xint)
    eval_heri = np.zeros(len(xeval))

    # Calculating the value of the interpolation
    for k in range(len(xeval)):
        eval_heri[k] = eval_hermite(xeval[k],xint,yint,ypint,n[i])
    eval_her.append(eval_heri)

    # Plotting results
    plt.figure()
    plt.plot(xeval,eval_heri,label = "p(x)")
    plt.scatter(xint,yint,label = "Interplation nodes")
    plt.plot(xeval,feval,label = "f(x)")
    plt.title(f"N = {n[i]}")
    plt.legend()
    plt.grid()

# # Creating the error plot
err = []
plt.figure()
for j in range(len(n)):
    err.append(np.log10(np.abs(eval_her[j] - feval)))
    plt.plot(xeval,err[j],label = f"n = {n[j]}")

plt.grid()
plt.xlabel("x")
plt.ylabel("log(error)")
plt.title("Interpolation error for different values of n: Hermite")
plt.legend()



# Solve using Natural Cubic Spline ===========================================================

eval_nat = []
for i in range(len(n)):
    xint = getXchev(n[i]+1,xmin,xmax)
    yint = f(xint)
    eval_nati = np.zeros(len(xeval))

    (M,C,D) = create_natural_spline(yint,xint,n[i])
    eval_nati = eval_cubic_spline(xeval,1000,xint,n[i],M,C,D)
    eval_nat.append(eval_nati)

    plt.figure()
    plt.plot(xeval,eval_nati,label="p(x)")
    plt.plot(xeval,feval,label = "f(x)")
    plt.scatter(xint,yint,label="Interpolation Nodes")
    plt.title(f"Natural cubic spline: n = {n[i]}")
    plt.legend()
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")

    

err = []
plt.figure()
for j in range(len(n)):
    err.append(np.log10(np.abs(eval_nat[j] - feval)))
    plt.plot(xeval,err[j],label = f"n = {n[j]}")

plt.grid()
plt.xlabel("x")
plt.ylabel("log(error)")
plt.title("Interpolation error for different values of n: Natural Cubic Spline")
plt.legend()



## CLAMPED SPLINE =========================================================================

eval_clam = []
fp0 = fp(xmin)
fpN = fp(xmax)
for i in range(len(n)):
    xint = getXchev(n[i]+1,xmin,xmax)
    yint = f(xint)
    eval_clami = np.zeros(len(xeval))

    (M,C,D) = create_clamped_spline(yint,xint,n[i],fp0,fpN)
    eval_clami = eval_cubic_spline(xeval,1000,xint,n[i],M,C,D)
    eval_clam.append(eval_clami)

    plt.figure()
    plt.plot(xeval,eval_clami,label="p(x)")
    plt.plot(xeval,feval,label = "f(x)")
    plt.scatter(xint,yint,label="Interpolation Nodes")
    plt.title(f"Clamped cubic spline: n = {n[i]}")
    plt.legend()
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")

    

err = []
plt.figure()
for j in range(len(n)):
    err.append(np.log10(np.abs(eval_clam[j] - feval)))
    plt.plot(xeval,err[j],label = f"n = {n[j]}")

plt.grid()
plt.xlabel("x")
plt.ylabel("log(error)")
plt.title("Interpolation error for different values of n: Clamped Cubic Spline")
plt.legend()
plt.show()