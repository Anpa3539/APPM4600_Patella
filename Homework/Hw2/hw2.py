import numpy as np
import matplotlib.pyplot as plt
import random

# Problem 1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A = np.array([[0.5, 0.5],[0.5*(1+1e-10), 0.5*(1-1e-10)]])
evals, evecs = np.linalg.eig(A)
print(evals)
print("Condition number is: ", np.abs(evals[0]/evals[1]))


b1 = 1
b2 = 1

Db1 = 1.54*10**(-5) + 1
Db2 = 8.71*10**(-5) + 1

Dx1 = (1-10**10)*Db1 + 10**10*Db2
Dx2 = (1+10**10)*Db1 - 10**10*Db2

err = (np.abs(np.sqrt(2) - np.sqrt(Dx1**2 +Dx2**2))/np.sqrt(2))
print(err)

print("Delta x= ")
print(Dx1)
print(Dx2)


# Problem 3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x = np.linspace(0.0001, 20, 100)
K = np.abs(x)*np.exp(x)/(np.exp(x)-1)

plt.plot(x,K)
plt.xlabel('x')
plt.ylabel('Condition Number')
plt.grid()

x0 = 9.999999995000000*10**(-10)
f = np.exp(x0)-1
print(f)

def f1(x):
    return x + x**2/2

f2 = f1(x0)
print(f2)


# Problem 4~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Python practice a
# Create a vector t from 0:pi/30:pi, and create a vector y = cos(t). Evaluate the sum. 


y = np.cos(t)
s = t*y
S = np.sum(s)
print('The sum is: ',S)


# Plotting

# Constants for the first figure
theta = np.linspace(0,2*np.pi,100)
R = 1.2
dr = 0.1
f = 15
p = 0

# Calculating x, y
x = R*(1+dr*np.sin(f*theta + p))*np.cos(theta)
y = R*(1+dr*np.sin(f*theta + p))*np.sin(theta)

# Creating the figure
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')

# Creating the several graphs for i in [1,10]
x1 = []
x2 = []

i = np.linspace(1,10,10)

plt.figure()
plt.xlabel('x')
plt.ylabel('y')

dr = 0.05

# Iterating through all of the values and plotting
for i in i:
    R = i
    f = 2+i
    p = random.uniform(0,2)
    a = R*(1+dr*np.sin(f*theta + p))*np.cos(theta)
    b = R*(1+dr*np.sin(f*theta + p))*np.sin(theta)
    
    np.append(x1,a)
    np.append(x1,b)
    plt.plot(a,b)
    

