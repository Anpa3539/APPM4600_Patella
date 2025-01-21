
import matplotlib.pyplot as plt
import numpy as np

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