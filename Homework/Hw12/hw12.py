import numpy as np

## Problem 2: #########################################################
A = np.array([[12,10,4],[10,8,-5],[4,-5,3]])
a1 = np.array([10,4])

e1 = np.array([1,0])
I = np.eye(2)

u = a1 + np.linalg.norm(a1)*e1
u = u/np.linalg.norm(u)

Q = I - 2*np.outer(u,u)

H = np.eye(3)
H[1,1] = Q[0,0]
H[1,2] = Q[0,1]
H[2,1] = Q[1,0]
H[2,2] = Q[1,1]
A_tridiag = H@A@np.transpose(H)

## PROBLEM 3: #########################################################
n = np.array([4,8,12,16,20])

def hilbert(n):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i,j] = 1/((i+1)+(j+1)-1)

    return(A)

def power_method(A,x0,nmax,tol,verb=False):
    # A - matrix to do power method on
    # x0 - initial guess of eigenvector
    # n - number of iterations
    iterations = 0
    scaling_factor = 1
    x1 = A@x0
    while iterations<nmax and np.linalg.norm(x1-x0)>tol:
        #
        if verb == True:
            print("xn = ",x0," n = ",iterations)
        x0 = x1
        x1 = A@x0
        scaling_factor = np.max(x1)
        x1 = x1/scaling_factor #Implementing scaling
        iterations +=1
    return (A,scaling_factor,iterations)



A = np.array([[1,2,0],[-2,1,2],[1,3,1]])
nmax = 100
tol = 1e-6

for i in range(len(n)):
    h = hilbert(n[i])
    x0 = np.ones((n[i],1))
    eigenvalue = np.zeros((len(n),1))
    _,eigenvalue[i],iterations = power_method(h,x0,nmax,tol)
    print("n = ",n[i],", lambda = ",eigenvalue[i], ", using ",iterations," iterations")

# Find the smallest eigenvalue
n = 16
h = hilbert(n)
x0 =  np.ones((n,1))
evec,eigenvalue,iterations = power_method(h,x0,nmax,tol)
B = h - eigenvalue*np.eye(n)
evec,eigenvalue,iterations = power_method(B,x0,nmax,tol)

print("Smallest eigenvalue")
print("n = ",n,", lambda = ",eigenvalue, ", using ",iterations," iterations")

# Finding the true eigenvalues
eigenvalue,eigenvector = np.linalg.eig(h)
print(eigenvalue)

# Trying again with inverse method
h_i = np.linalg.inv(h)
evec,eigenvalue,iterations = power_method(h_i,x0,nmax,tol)
print("n = ",n,", lambda_max = ",eigenvalue, ", using ",iterations," iterations")
print("n = ",n,", lambda_min = ",eigenvalue**(-1), ", using ",iterations," iterations")


## CHECKING IF IT MATCHES THE THEOREM
E = 0.001 *np.eye(16)
x0 = np.ones(16)
B = h + E
B_i = np.linalg.inv(B)
_,lambda_min,_ = power_method(B_i,x0,nmax,tol)
norm_E = np.linalg.norm(E,2)


if norm_E>np.abs(lambda_min**(-1)-eigenvalue**(-1)):
    print("Theorem is satisfied")
else:
    print("Theorem is not satisfied")

print(lambda_min**(-1)-eigenvalue**(-1))
print(norm_E)

## Example of the power method not working
A = np.array([[-15,6],[-6,-15]])
#A = np.ones((3,3))
#eigs = np.array([1,1,0.99999999])
#A = np.diag(eigs)
print(A)
x0 = np.ones((2,1))
evec,eigenvalue,iterations = power_method(A,x0,nmax,tol)
print(eigenvalue)






