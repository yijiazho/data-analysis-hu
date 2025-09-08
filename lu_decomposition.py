import numpy as np
import copy
from scipy.linalg import lu
import matplotlib.pyplot as plt


# Assume no pivoting
def lu_decomposition(A):
    n = len(A)
    
    L = np.eye(n)
    U = np.zeros((n, n))
    
    for i in range(n):
        # 0 <= i < j < n
        # Fill in the U[i][j]
        for j in range(i, n):
            sum = 0
            for k in range(i):
                sum += L[i][k] * U[k][j]
            U[i][j] = A[i][j] - sum
        
        # Fill in the L[j][i]
        for j in range(i+1, n):
            sum = 0
            for k in range(i):
                sum += L[j][k] * U[k][i]
            L[j][i] = (A[j][i] - sum) / U[i][i]
    
    return L, U


A = np.array([[7, 2, -3], 
              [2, 5, -3], 
              [1, -1, 6]])
print(A)

P, L, U = lu(A)

print(P)
print(L)
print(U)

print(np.matmul(L, U))

L1, U1 = lu_decomposition(A)

print(L1)
print(U1)
print(np.matmul(L1, U1))

def gauss_seidel(A, b, tolerance=0.05, max_iteration=100, lamb=1):
    n = len(A)
    x_old = np.zeros(n)
    errors = []
    error = 1
    iteration = 0
    
    while error > tolerance and iteration < max_iteration:
        x_new = np.zeros(n)
        max_error = 0
        for i in range(n):
            sum1 = 0
            sum2 = 0
            for j in range(n):
                if j < i:
                    sum1 += A[i][j] * x_new[j]
                elif j > i:
                    sum2 += A[i][j] * x_old[j]
            new_value = (b[i] - sum1 - sum2) / A[i][i]
            x_new[i] = lamb * new_value + (1 - lamb) * x_old[i]
            max_error = max(max_error, abs((x_new[i] - x_old[i])/ x_new[i]))
        error = max_error
        errors.append(error)
        print(
            f"Iteration {iteration + 1}: {x_new}"
            f"Relative Error = {100 * error}%"
        )
        x_old = x_new
        iteration += 1
        
    return x_new, errors
        
AA = np.array([[6, -2, 1], 
              [5, 10, 1], 
              [-3, 1, 15]], dtype='d')

b = np.array([5, 28, 44], dtype='d')
_, errors_1 = gauss_seidel(AA, b)
_, errors_2 = gauss_seidel(AA, b, lamb=0.95)


plt.figure(figsize=(10, 6))
plt.plot(errors_1, label="lambda 1.0")
plt.plot(errors_2, label="lambda 0.95")
plt.xlabel('Iteration')
plt.ylabel('Log10(Error)')
plt.title('Convergence of Gauss-Seidel with Different Relaxation Parameters')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()