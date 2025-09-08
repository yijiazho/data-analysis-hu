import numpy as np
import math

array = np.array(
    [0.9, 1.42, 1.3, 1.55, 1.63, 
     1.32, 1.35, 1.47, 1.95, 1.66, 
     1.96, 1.47, 1.92, 1.35, 1.05, 
     1.85, 1.74, 1.65, 1.78, 1.71, 
     2.29, 1.82, 2.06, 2.14, 1.27]
)

def mean(array):
    return sum(array) / len(array)

print(mean(array))
print(np.mean(array))

def std_deviation(array):
    m = mean(array)
    sum = 0
    for n in array:
        sum += (n - m) ** 2
    variance = sum / len(array)
    return math.sqrt(variance)

print(std_deviation(array))
print(np.std(array))

def variance(array):
    m = mean(array)
    sum = 0
    for n in array:
        sum += (n - m) ** 2
    return sum / len(array)


print(variance(array))
print(np.var(array))

def coefficient_variance(array):
    return (std_deviation(array) / mean(array)) * 100

print(coefficient_variance(array))


def confidence_interval_95(array):
    z_value = 1.96
    margin_of_error = z_value * (std_deviation(array) / math.sqrt(len(array)))
    m = mean(array)
    return (m - margin_of_error, m + margin_of_error)

print(confidence_interval_95(array))

import matplotlib.pyplot as plt

bins = np.linspace(0.6, 2.4, num=10)
plt.hist(array, bins=bins, edgecolor='black')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Dataset')

# Show the plot
plt.show()

x = np.array([0, 1, 2.5, 3, 4.5, 5, 6])
y = np.array([26, 15.5, 5.375, 3.5, 2.375, 3.5, 8])

def fdd(x, y):
    n = len(x)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = (diff_table[i + 1][j - 1] - diff_table[i][j - 1]) / (x[i + j] - x[i])
    
    return diff_table

def newton_interpolation(x, y, value):
    diff_table = fdd(x, y)
    n = len(x)
    # Construct fdd
    result = diff_table[0, 0]
    
    # Construct the Newton polynomial using divided differences
    for i in range(1, n):
        term = diff_table[0, i]
        for j in range(i):
            term *= (value - x[j])
        result += term
    
    return result

print(newton_interpolation(x, y, 3.5))