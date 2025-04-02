import numpy as np


# Problem 1: Solve system with least squares
A = np.array([[1,0],[0,1],[0,1]])
y = np.array([[1],[1],[0]])

G = np.transpose(A)@A
b = np.transpose(A)@y

x = np.linalg.inv(G)@b
print(x)

# Problem 2: Solve using least squares
A = np.array([[1,3],[12,-2],[20,0],[6,21]])
