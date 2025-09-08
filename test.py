import numpy as np
import math

ITERATION_LIMIT = 1000

# initialize the matrix
A = np.array(
    [
        [6, -2, 1],
        [5, 10, 1],
        [-3, 1, 15],
    ], dtype='d'
)
# initialize the RHS vector
b = np.array([5, 28, 44], dtype='d')

print("System of equations:")
for i in range(A.shape[0]):
    row = [f"{A[i,j]:3g}*x{j+1}" for j in range(A.shape[1])]
    print("[{0}] = [{1:3g}]".format(" + ".join(row), b[i]))

x = np.zeros_like(b, np.float_)
for it_count in range(1, ITERATION_LIMIT):
    x_new = np.zeros_like(x, dtype=np.float_)
    print(f"Iteration {it_count}: {x}")
    for i in range(A.shape[0]):
        s1 = np.dot(A[i, :i], x_new[:i])
        s2 = np.dot(A[i, i + 1 :], x[i + 1 :])
        x_new[i] = (b[i] - s1 - s2) / A[i, i]
    if np.allclose(x, x_new, rtol=1e-8):
        break
    x = x_new

print(f"Solution: {x}")
error = np.dot(A, x) - b
print(f"Error: {error}")