# import libraries
import numpy as np


def driver():
    # test functions
    def f1(x):
        return np.sqrt(10/(x+4))

    Nmax = 100
    tol = 1e-6

    # test f1 '''
    x0 = 1.5
    [xstar, ier, p] = fixedpt(f1, x0, tol, Nmax)
    print('the approximate fixed point is:', xstar)
    print('f1(xstar):', f1(xstar))
    print('Error message reads:', ier)
    print(p)

	# Calculating the coefficients
    p_actual = 1.3652300134140976
    n = 2
    alpha = np.log(np.abs(p[n+1]-p_actual)/np.abs(p[n]-p_actual))/np.log(np.abs((p[n]-p_actual)/np.abs(p[n-1]-p_actual)))
    l = np.abs(p[n+1]-p_actual)/np.abs(p[n]-p_actual)**alpha

    print("Alpha is ",alpha)
    print('Lambda is ', l)

# Aitken Method
    pn_hat = aitken(p,tol,Nmax)
    print("Pn_hat is ", pn_hat)

# define routines
def fixedpt(f, x0, tol, Nmax):

    count = 0
    p=[]
    while (count < Nmax):
        p.append(float(x0))
        count = count + 1
        x1 = f(x0)
        if (abs(x1 - x0) < tol):
            xstar = x1
            ier = 0

            return [xstar, ier, p]
        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, p]

# Coding Aitken Method
def aitken(p,tol,Nmax):
    n = 0
    pn_hat = []
    while n<(Nmax -2):
        pn_hat1 = p[n]-((p[n+1]-p[n])**2)/(p[n+2]-2*p[n+1]+p[n])
        pn_hat.append(pn_hat1)
        n = n+1
        print("run ", n)
        return pn_hat

driver()



