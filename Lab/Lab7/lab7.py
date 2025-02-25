import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def vandermonde(x):
    n = len(x)
    v = np.empty((n,n))
    # i = row, j = col
    for i in range(n):
        for j in range(n):
            v[i,j] = x[i]**j
    return v

def interp_coefficients(v,y):
    a = np.linalg.solve(v,y)
    return a

def solve_coeff(a,xq):
    p = []
    n = len(a)
    m = len(xq)
    for i in range(m):
        pn = 0
        for j in range(n):
            pn = a[j]*xq[i]**j + pn
        p.append(pn)
    return(p)

f = lambda x: solve_coeff(a,x)
        
n = 15
x = [100*ran.random() for _ in range(n)]
y = [100*ran.random() for _ in range(n)]
xq = np.array([1,np.pi,45,0.2])

xempty = np.linspace(0,100,1000)
#x = np.array([1,3,5,14])
#y = np.array([0,2,-1,-6])

v = vandermonde(x)
a = interp_coefficients(v,y)
p = solve_coeff(a,xq)
print(p)

plt.figure()
plt.scatter(x,y)
plt.plot(xempty,f(xempty))

    


def eval_monomial(xeval,coef,N,Neval):
    yeval = coef[0]*np.ones(Neval+1)
# print('yeval = ', yeval)
    for j in range(1,N+1):
        for i in range(Neval+1):
# print('yeval[i] = ', yeval[i])
# print('a[j] = ', a[j])
# print('i = ', i)
# print('xeval[i] = ', xeval[i])
            yeval[i] = yeval[i] + coef[j]*xeval[i]**j
    return yeval

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
  
    


#''' create divided difference matrix'''
def dividedDiffTable(x, y, n):
 
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                                     (x[j] - x[i + j]));
    return y;
    
def evalDDpoly(xval, xint,y,N):
    #''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)
    
    ptmp[0] = 1.
    for j in range(N):
      ptmp[j+1] = ptmp[j]*(xval-xint[j])
     
    #'''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
       yeval = yeval + y[0][j]*ptmp[j]  

    return yeval


def vandermonde(x):
    n = len(x)
    v = np.empty((n,n))
    # i = row, j = col
    for i in range(n):
        for j in range(n):
            v[i,j] = x[i]**j
    return v

       


f = lambda x: 1/(1+(10*x)**2)

N = 3
    #''' interval'''
a = -1
b = 1
   
   
    #''' create equispaced interpolation nodes'''
xint = np.linspace(a,b,N+1)
    
    #''' create interpolation data'''
yint = f(xint)
    
    #''' create points for evaluating the Lagrange interpolating polynomial'''
Neval = 1000
xeval = np.linspace(a,b,Neval+1)
yeval_l= np.zeros(Neval+1)
yeval_dd = np.zeros(Neval+1)
  
    #'''Initialize and populate the first columns of the 
     #divided difference matrix. We will pass the x vector'''
y = np.zeros( (N+1, N+1) )
     
for j in range(N+1):
    y[j][0]  = yint[j]

y = dividedDiffTable(xint, y, N+1)
    #''' evaluate lagrange poly '''
for kk in range(Neval+1):
    yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
    yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)
    
   # ''' create vector with exact values'''
fex = f(xeval)
       

plt.figure()    
plt.plot(xeval,fex)
plt.plot(xeval,yeval_l) 
plt.plot(xeval,yeval_dd)

plt.figure() 
err_l = abs(yeval_l-fex)
err_dd = abs(yeval_dd-fex)
plt.semilogy(xeval,err_l,label='lagrange')
plt.semilogy(xeval,err_dd,label='Newton DD')
plt.legend()
plt.show()

a = -1
b = 1
N = 4
f = lambda x: 1/(1+(10*x)**2)

# interpolating nodes
xint = np.linspace(a, b, N+1)
yint = f(xint)

V = vandermonde(xint)
Vinv = np.linalg.inv(V)
coef = Vinv @ yint

# evaluation points
Neval = 1000
xeval = np.linspace(a, b, Neval+1)
yeval = f(xeval)

plt.plot(yeval, label='f(x)')
y_mon = eval_monomial(xeval, coef, N, Neval)
plt.plot(y_mon, label='monomial')

y_lag = np.zeros(Neval+1)
for i in range(Neval+1):
    y_lag[i] = eval_lagrange(xeval[i], xint, yint, N)
plt.plot(y_lag, label='lagrange')

y = np.zeros((N+1,N+1))
y[:,0] = yint
y_ndd = np.zeros(Neval + 1)
for i in range(Neval + 1):
    y_ndd[i] = evalDDpoly(xeval[i], xint, dividedDiffTable(xint, y, N+1), N)
plt.plot(y_ndd, label='ddpoly')

plt.legend()
plt.show()

# FOR LOOP FOR ALL N VALUES =========================================================
N = np.array([2,3,4,5,6,7,8,9,10])
for N in N:
    xint = np.linspace(a, b, N+1)
    yint = f(xint)

    V = vandermonde(xint)
    Vinv = np.linalg.inv(V)
    coef = Vinv @ yint

# evaluation points
    Neval = 1000
    xeval = np.linspace(a, b, Neval+1)
    yeval = f(xeval)
    
    plt.plot(xeval,yeval, label='f(x)')
    y_mon = eval_monomial(xeval, coef, N, Neval)
    plt.plot(xeval,y_mon, label='monomial')

    y_lag = np.zeros(Neval+1)
    for i in range(Neval+1):
        y_lag[i] = eval_lagrange(xeval[i], xint, yint, N)
    plt.plot(xeval,y_lag, label='lagrange')

    y = np.zeros((N+1,N+1))
    y[:,0] = yint
    y_ndd = np.zeros(Neval + 1)
    for i in range(Neval + 1):
        y_ndd[i] = evalDDpoly(xeval[i], xint, dividedDiffTable(xint, y, N+1), N)
    plt.plot(xeval,y_ndd, label='ddpoly')

    plt.legend()
    plt.show()

    fex = f(xeval)
    plt.figure() 
    err_l = abs(yeval_l-fex)
    err_dd = abs(yeval_dd-fex)
    plt.semilogy(xeval,err_l,label='lagrange')
    plt.semilogy(xeval,err_dd,label='Newton DD')
    plt.semilogy(xeval,np.abs(fex-yeval))
    plt.legend()
    plt.show()
    




       
