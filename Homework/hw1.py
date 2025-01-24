import matplotlib.pyplot as plt
import numpy as np

# Question 1
start = 1.920
end = 2.080
step = 0.001

n = int((end-start)/step) +1
x = np.linspace(1.920,2.080,n)

# i) Plot using coefficients
p1 = x**9 -18*x**8 +144*x**7 -672*x**6 +2016*x**5 -4032*x**4 +5376*x**3 -4608*x**2 +2304*x-512
plt.plot(x, p1)
plt.xlabel("x")
plt.ylabel("p1(x)")
plt.title("Polynomial Plot")
plt.grid()
plt.show()

# ii) Plot using the factored form 
p2 = (x-2)**9
plt.plot(x, p2)
plt.xlabel("x")
plt.ylabel("p2(x)")
plt.title("Polynomial Plot")
plt.grid()
plt.show()


##################################################
# Question 5
def trig(x,delta):
    return -2*np.sin(x + delta/2) * np.sin(delta/2)

def trig1(x,delta):
    return np.cos(x+delta)-np.cos(x)

def trig2(x,delta):
    # Assuming |cos(\xi)|\leq 1
    return (delta*np.sin(x) + delta**2/2)
    
x1 = np.pi
x2 = 10**6

delta = np.logspace(-16,0,17)

# Evaluating the expression using no subtraction
f1 = trig(x1,delta)
f2 = trig(x2,delta)

ff1 = trig1(x1,delta)
ff2 = trig1(x2,delta)

fa1 = trig2(x1,delta)
fa2 = trig2(x2,delta)

error1 = -(ff1-f1)
error2 = -(ff2-f2)

error3 = -(fa1-f1)
error4 = -(fa2-f2)

# Create figure for x1, x2
plt.plot(delta,error1)
plt.xlabel('delta')
plt.ylabel('f(x+delta)')
plt.xscale('log')
plt.title('Difference between expressions for different error values')
plt.show()

plt.plot(delta,error2)
plt.xlabel('delta')
plt.ylabel('f(x+delta)')
plt.xscale('log')
plt.title('Difference between expressions for different error values')
plt.show()

# Plotting error for my algorithm, with Taylor series
plt.plot(delta,error3)
plt.xlabel('delta')
plt.ylabel('f(x+delta)')
plt.xscale('log')
plt.title('Difference between expressions for different error values')
plt.show()

plt.plot(delta,error4)
plt.xlabel('delta')
plt.ylabel('f(x+delta)')
plt.xscale('log')
plt.title('Difference between expressions for different error values')
plt.show()