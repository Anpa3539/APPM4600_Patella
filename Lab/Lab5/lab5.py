import numpy as np
import matplotlib.pyplot as plt


## NORMAL METHODS ###############################################################
def newtons(f,fp,x0,Nmax,tol):
    N = 0
    if fp(x0) == 0:
        print("Error: f'(x)=0. Method Failed. Try another x0 or method.")
        print("Method failed when n = ", N, ", x = ", x0)
    x1 = x0 - f(x0)/fp(x0)
    N = 1
    seq = []

    while np.abs(x1-x0)>tol and N<Nmax:
        seq.append(x0)
        x0 = x1
        x1 = x1 = x0 - f(x0)/fp(x0)
        N = N+1

        if fp(x1) == 0:
            print("Error: f'(x)=0. Method Failed. Try another x0 or method.")
            print("Method failed when n = ", N,", x = ", x1)
            break
    return x1,N,seq

def bisection(f, a, b, tol):

    fa = f(a)
    fb = f(b);

    if (fa * fb > 0):
        ier = 1
        astar = a

        return [astar, ier]

    #   verify end points are not a root
    if (fa == 0):
        astar = a
        ier = 0

        return [astar, ier]

    if (fb == 0):
        astar = b
        ier = 0

        return [astar, ier]

    count = 0
    d = 0.5 * (a + b)
    while (abs(d - a) > tol):
        fd = f(d)
        print("d = ",d,", n = ", count)
        if (fd == 0):
            astar = d
            ier = 0
            print(astar)
            return [astar, ier]
        if (fa * fd < 0):
            b = d
        else:
            a = d
            fa = fd
        d = 0.5 * (a + b)
        count = count + 1
    #      print('abs(d-a) = ', abs(d-a))

    astar = d

    ier = 0

    return [astar, ier]

## MY HYBRID METHOD #################################################################
def hybrid(f,fp,a,b,Nmax,tol):
    basin = 1e-1
    x0,ier = bisection(f, a, b, basin)

    root,N,seq = newtons(f,fp,x0,Nmax,tol)
    return root

# Functions
f = lambda x: np.exp(x**2+7*x-30)-1
fp = lambda x: (2*x + 7)*np.exp(x**2+7*x-30)
fpp = lambda x: 2*f(x) + ((2*x+7)**2)*f(x)

## TRYING THE METHODS ##############################################################
# Part 1: Using just bisection
a = 2
b = 4.5
tol = 1e-9
root_bisection,ier = bisection(f, a, b, tol)

# Pat 2: Newtons Method only
x0 = 4.5
Nmax = 100
root_newtons,n,seq = newtons(f,fp,x0,Nmax,tol)
print(seq)
print("N = ", n)

# Part 3: Hybrid Method
root_hybrid = hybrid(f,fp,a,b,Nmax,tol)

## CREATING A NEW BETTER HYBRID METHOD #################################################
# Design a hybrid method that has a smarter convergence criteria
g = lambda x: x - f(x)/fp(x)


def bisection2(f,fp,fpp, a, b, tol_stop):
    gp = lambda x: fpp(x)/(fp(x)**2)
    fa = f(a)
    fb = f(b);
    if (fa * fb > 0):
        ier = 1
        astar = a
        return [astar, ier]
    #   verify end points are not a root
    if (fa == 0):
        astar = a
        ier = 0
        return [astar, ier]
    if (fb == 0):
        astar = b
        ier = 0
        return [astar, ier]
    count = 0
    d = 0.5 * (a + b)
    while (gp(a) > tol_stop and gp(b)>tol_stop):
        fd = f(d)
        print("d = ",d,", n = ", count)
        if (fd == 0):
            astar = d
            ier = 0
            print(astar)
            return [astar, ier]
        if (fa * fd < 0):
            b = d
        else:
            a = d
            fa = fd
        d = 0.5 * (a + b)
        count = count + 1
    #      print('abs(d-a) = ', abs(d-a))
    astar = d
    ier = 0
    return [astar, ier,count]



def hybrid2(f,fp,fpp,a,b,Nmax,tol,tol_stop):
    
    x0,ier,count = bisection2(f,fp,fpp, a, b, tol_stop)
    print(count)
    root,N,seq = newtons(f,fp,x0,Nmax,tol)
    iterations = N + count
    return root, iterations


## USING THE NEW HYBRID METHOD#####################################################

# Trying the new hybrid method
tol_stop = 1e-3
root_hybrid2, iterations = hybrid2(f,fp,fpp,a,b,Nmax,tol,tol_stop)
print("root at x = ",root_hybrid2,", num iterations: ",iterations)


## VALIDATING WITH A NEW FUNCTION ##################################################
# Validating with other functions
f = lambda x: 2+x-x**4 + np.exp(x)
fp = lambda x: 1 - 4*x**3 + np.exp(x)
fpp = lambda x: -12*x**2 + np.exp(x)

tol_stop = 1e-3
tol = 1e-9
Nmax = 100
a = 4
b = 10

root_hybrid2_1, iterations_1 = hybrid2(f,fp,fpp,a,b,Nmax,tol,tol_stop)
print("root at x = ",root_hybrid2_1,", num iterations: ",iterations_1)



 