from scipy import special
import matplotlib.pyplot as plt
import numpy as np

## FUNCTIONS #######################################################
def bisection(f, a, b, tol):

    fa = f(a)
    fb = f(b);
    count = 1

    if (fa * fb > 0):
        ier = 1
        astar = a

        return [astar, ier,count]

    #   verify end points are not a root
    if (fa == 0):
        astar = a
        ier = 0

        return [astar, ier,count]

    if (fb == 0):
        astar = b
        ier = 0

        return [astar, ier,count]

    
    d = 0.5 * (a + b)
    while (abs(d - a) > tol):
        fd = f(d)
        #print("d = ",d,", n = ", count)
        if (fd == 0):
            astar = d
            ier = 0
            print(astar)
            return [astar, ier,count]
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

def newton_method1(f,df,x0,tol,nmax,verb=False):
    #newton method to find root of f starting at guess x0

    #Initialize iterates and iterate list
    xn=x0;
    rn=np.array([x0]);
    # function evaluations
    fn=f(xn); dfn=df(xn);
    nfun=2; #evaluation counter nfun
    dtol=1e-10; #tolerance for derivative (being near 0)

    if abs(dfn)<dtol:
        #If derivative is too small, Newton will fail. Error message is
        #displayed and code terminates.
        if verb:
            fprintf('\n derivative at initial guess is near 0, try different x0 \n');
    else:
        n=0;
        if verb:
            print("\n|--n--|----xn----|---|f(xn)|---|---|f'(xn)|---|");

        #Iteration runs until f(xn) is small enough or nmax iterations are computed.

        while n<=nmax:
            if verb:
                print("|--%d--|%1.8f|%1.8f|%1.8f|" %(n,xn,np.abs(fn),np.abs(dfn)));

            pn = - fn/dfn; #Newton step
            if np.abs(pn)<tol or np.abs(fn)<2e-15:
                break;

            #Update guess adding Newton step
            xn = xn + pn;

            # Update info and loop
            n+=1;
            rn=np.append(rn,xn);
            dfn=df(xn);
            fn=f(xn);
            nfun+=2;

        r=xn;

        if n>=nmax:
            print("Newton method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)));
        else:
            print("Newton method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)));

    return (r,rn,nfun)

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

def secantPrint(f,a,b,Nmax,tol):
    p = []
    
    x0 = a
    x1 = b
    n = 0
    print("x0---------,---|x0-x1|-----,log(|x0-x1|)--,N----")
    while np.abs(x0-x1)>tol and n<Nmax:
        print("%0.9f" %x0,", %0.9f" %np.abs(x0-x1),", %0.9f" %np.log10(np.abs(x0-x1)),", ",n)
        p.append(x0)
        x0 = x1
        x1 = x0 - (f(x0)*((x1)-(x0)))/(f(x1)-f(x0))
        n = n+1

    return x0,n,p

def secant_method(f,x0,x1,tol,nmax,verb=False):
    #secant (quasi-newton) method to find root of f starting with guesses x0 and x1
    p = []
    #Initialize iterates and iterate list
    xnm=x0; xn=x1;
    rn=np.array([x1]);
    # function evaluations
    fn=f(xn); fnm=f(xnm);
    msec = (fn-fnm)/(xn-xnm);
    nfun=2; #evaluation counter nfun
    dtol=1e-10; #tolerance for derivative (being near 0)

    if np.abs(msec)<dtol:
        #If slope of secant is too small, secant will fail. Error message is
        #displayed and code terminates.
        if verb:
            fprintf('\n slope of secant at initial guess is near 0, try different x0,x1 \n');
    else:
        n=0;
        if verb:
            print("\n|--n--|----xn----|---|f(xn)|---|---|msec|---|");

        #Iteration runs until f(xn) is small enough or nmax iterations are computed.

        while n<=nmax:
            
            if verb:
                #print("|--%d--|%1.8f|%1.8f|%1.8f|" %(n,xn,np.abs(fn),np.abs(msec)));
                a=1+1
                
            pn = - fn/msec; #Secant step
            if np.abs(pn)<tol or np.abs(fn)<2e-15:
                break;

            #Update guess adding Newton step, update xn-1
            p.append(xn)
            xnm = xn; #xn-1 is now xn
            xn = xn + pn; #xn is now xn+pn

            # Update info and loop
            print("%0.9f" %xn,"& %0.9f" %np.abs(pn),"& %0.9f" %np.log10(np.abs(pn)),"& ",n)
            n+=1;
            rn=np.append(rn,xn);
            fnm = fn; #Note we can re-use this function evaluation
            fn=f(xn); #So, only one extra evaluation is needed per iteration
            msec = (fn-fnm)/(xn-xnm); # New slope of secant line
            nfun+=1;

        r=xn;
        
        if n>=nmax:
            print("Secant method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)));
        else:
            print("Secant method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)));

    return (r,rn,nfun,p)

def newtonsPrint(f,fp,x0,Nmax,tol):
    N = 0
    if fp(x0) == 0:
        print("Error: f'(x)=0. Method Failed. Try another x0 or method.")
        print("Method failed when n = ", N, ", x = ", x0)
    x1 = x0 - f(x0)/fp(x0)
    N = 1
    seq = []
    print("x0---------,---|x0-x1|-----,log(|x0-x1|)--,N----")
    while np.abs(x1-x0)>tol/2 and N<Nmax:
        print("%0.9f" %x0,"& %0.9f" %np.abs(x0-x1),"& %0.9f" %np.log10(np.abs(x0-x1)),"& ",N)
        seq.append(x0)
        x0 = x1
        x1 = x1 = x0 - f(x0)/fp(x0)
        N = N+1

        if fp(x1) == 0:
            print("Error: f'(x)=0. Method Failed. Try another x0 or method.")
            print("Method failed when n = ", N,", x = ", x1)
            break
    return x1,N,seq


## PROBLEM 1 ########################################################################
# Problem 1:
Ti = 20
Ts = -15
alpha = 0.138e-6
T = lambda x,t: special.erf(x/(2*np.sqrt(alpha*t)))*(Ti-Ts)+Ts

# Find the value of x when t=60 days

# convert t to seconds so it matches alpha units
ta = 60*3600*24 #s
# solve T = Ta ==> T-Ta = 0
f = lambda x: T(x,ta)
fp = lambda x: 2/(2*np.sqrt(alpha*ta*np.pi))*np.exp(-x**2)*(Ti-Ts)



x = np.linspace(0,1,100)
yaxis = np.zeros((100,1))
plt.plot(x,f(x))
plt.plot(x,yaxis)
plt.xlabel('x [m]')
plt.ylabel('Temperature [C]')


a = 0
b = 1
x0 = 0.01
tol = 1e-10
Nmax = 100
depth,ier,n = bisection(f, a, b, tol)
print("The depth to not freeze is ", depth, " m")
print("Took ", n, " iterations. ")

depth,N,seq = newtons(f,fp,x0,Nmax,tol)
print("The depth to not freeze is ", depth, " m")
print("Took ", N, " iterations. ")

depth,N,seq = newtons(f,fp,b,Nmax,tol)
print("The depth to not freeze is ", depth, " m")
print("Took ", N, " iterations. ")


## PROBLEM 4 #######################################################################
# Problem 4: 
f = lambda x: np.exp(3*x)-27*x**6+27*x**4*np.exp(x)-9*x**2*np.exp(2*x)
fp = lambda x: 3*np.exp(3*x)-162*x**5+27*x**4*np.exp(x)+108*x**3*np.exp(x)-18*x**2*np.exp(2*x)-18*x*np.exp(2*x)
fpp = lambda x: 9*np.exp(3*x) - 810*x**4 + 27*x**4*np.exp(x) + 216*x**3*np.exp(x) + 324*x**2*np.exp(x) - 18*np.exp(2*x) - 72*x*np.exp(2*x) - 36*x**2*np.exp(2*x)
fppp = lambda x: 27*np.exp(3*x) - 3240*x**3 + 27*x**4*np.exp(x) + 648*x**3*np.exp(x) + 648*x**2*np.exp(x) + 648*x**2*np.exp(x) + 216*x**3*np.exp(x) - 36*np.exp(2*x) - 72*x*np.exp(2*x) - 72*np.exp(2*x) - 72*x*np.exp(2*x) - 72*x*np.exp(2*x) - 36*x**2*np.exp(2*x)
fpppp = lambda x: 81*np.exp(3*x) - 9720*x**2 + 27*x**4*np.exp(x) + 1296*x**3*np.exp(x) + 1296*x**2*np.exp(x) + 648*x**2*np.exp(x) + 648*x**2*np.exp(x) + 216*x**3*np.exp(x) - 72*np.exp(2*x) - 144*x*np.exp(2*x) - 72*np.exp(2*x) - 144*x*np.exp(2*x) - 72*x*np.exp(2*x) - 72*x*np.exp(2*x) - 72*x*np.exp(2*x) - 36*x**2*np.exp(2*x)


a = 3
b = 5

x0 = 3.5
Nmax = 100
tol =1e-25
# i) Newton's Method
rooti,Ni,p = newtonsPrint(f,fp,x0,Nmax,tol)
p_actual = rooti
n = 5

alpha = np.log(np.abs(p[n+1]-p_actual)/np.abs(p[n]-p_actual))/np.log(np.abs((p[n]-p_actual)/np.abs(p[n-1]-p_actual)))
l = np.abs(p[n+1]-p_actual)/np.abs(p[n]-p_actual)**alpha
print("For Newton's Method: alpha = ",alpha, ", lambda = ", l)
print(" ")

# ii) Modified Method from Class
# F(x) = f(x)/f'(x)
F = lambda x: f(x)/fp(x)
Fp = lambda x: 1 - f(x)*fpp(x)/(fp(x)**2)

rootii,Nii,p = newtonsPrint(F,Fp,x0,Nmax,tol)
p_actual = rootii
n = 2
print(fpppp(rootii))
alpha = np.log(np.abs(p[n+1]-p_actual)/np.abs(p[n]-p_actual))/np.log(np.abs((p[n]-p_actual)/np.abs(p[n-1]-p_actual)))
l = np.abs(p[n+1]-p_actual)/np.abs(p[n]-p_actual)**alpha
print("For 1st Modified Newton's Method: alpha = ",alpha, ", lambda = ", l)

# iii) Modified newton's from part 2


## PROBLEM 5 ########################################################################
# Problem 5
f = lambda x: x**6 - x - 1
fp = lambda x: 6*x**5 - 1

# using newtons:
x0 = 2
Nmax = 100
tol = 1e-9
print("Newton's Method")
xNewt,nNewt, seqNewt = newtonsPrint(f,fp,x0,Nmax,tol)

xNewt = xNewt*np.ones(len(seqNewt))
errNewt = np.log10(np.abs(seqNewt - xNewt))
plt.figure()
plt.plot(errNewt)
plt.xlabel('Number of Iterations')
plt.ylabel('log|x-alpha|')
plt.grid()

# Using Secant:
a = 2
b = 1
print("\nSecant Method")
xSec,nSec,lSec,seqSec = secant_method(f,a,b,tol,Nmax,True)
xSec = xSec*np.ones(len(seqSec))
errSec = np.log10(np.abs(seqSec - xSec))
plt.figure()
plt.plot(errSec)
plt.xlabel('Number of Iterations')
plt.ylabel('log|x-alpha|')
plt.grid()


# Creating the plots
r_N, xn_N,n_N = newton_method1(f,fp,2,1e-9,1000)
r_S, xn_S,n_S,p = secant_method(f,2,1,1e-9,1000)



plt.plot(range(8),np.abs(r_N-xn_N),label = "Newton")
plt.plot(range(8),np.abs(r_S-xn_S),label = "Secant")
plt.legend()
plt.xlabel('n')
plt.ylabel('Error')
plt.show
plt.grid()

plt.figure()
plt.loglog(np.abs(xn_N-r_N),np.abs(np.roll(xn_N,1)-r_N),label = "Newton")
plt.loglog(np.abs(xn_S-r_S),np.abs(np.roll(xn_S,1)-r_S),label = "Secant")
plt.xlabel("|x_n-r|")
plt.ylabel("|x_(n+1)-r|")
plt.legend()
plt.grid()

p_actual = r_N
p = xn_N
n = 2

alpha = np.log(np.abs(p[n+1]-p_actual)/np.abs(p[n]-p_actual))/np.log(np.abs((p[n]-p_actual)/np.abs(p[n-1]-p_actual)))
l = np.abs(p[n+1]-p_actual)/np.abs(p[n]-p_actual)**alpha
print("For 1st Modified Newton's Method: alpha = ",alpha, ", lambda = ", l)

p_actual = r_S
p = xn_S
n = 2

alpha = np.log(np.abs(p[n+1]-p_actual)/np.abs(p[n]-p_actual))/np.log(np.abs((p[n]-p_actual)/np.abs(p[n-1]-p_actual)))
l = np.abs(p[n+1]-p_actual)/np.abs(p[n]-p_actual)**alpha
print("For 1st Modified Newton's Method: alpha = ",alpha, ", lambda = ", l)



