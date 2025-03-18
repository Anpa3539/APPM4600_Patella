import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
from scipy.integrate import quad

f = lambda x: np.exp(x)
a = 1
b = 2
integral = quad(f,a,b)
print(integral)