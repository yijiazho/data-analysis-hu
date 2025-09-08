import math

true_value = 5 / 4 * (math.e) ** 6 + 1 / 4
print(true_value)


import numpy as np

def f(x):
    return x * math.e ** (2 * x)

def romberg_integration(f, a, b, n):
    """
    Perform Romberg integration for the given function

    Parameters:
        func: callable
        a: The lower limit
        b: The upper limit of integration.
        n: The maximum depth
    Returns:
        The integral and relative error
    """
    I = np.zeros((n, n))

    for j in range(n):
        num_intervals = 2 ** j
        h = (b - a) / num_intervals
        I[j, 0] = 0.5 * h * (f(a) + f(b) + 2 * sum(f(a + i * h) for i in range(1, num_intervals)))

    # Recursive formula for higher-order corrections
    for k in range(1, n):
        for j in range(n - k):
            I[j, k] = (4 ** k * I[j + 1, k - 1] - I[j, k - 1]) / (4 ** k - 1)

    return I[0, n - 1], abs(I[0, k - 1] - I[1, k - 2]) / I[0, k - 1] * 100

# Example usage
result, error = romberg_integration(f, 0, 3, 4)
print("Romberg integration result:", result)
print(f"Relative Error is: {error} %")


def integral(j, k):
    return (4 ** (k - 1) * integral(j + 1, k - 1) - integral(j, k - 1)) / (4 ** (k - 1) - 1)

def trapezoidal_rule(f, a, b, n):
    """
    Perform the trapezoidal rule

    Parameters:
        f: callable function
        n: The number of intervals.
        a: The lower limit
        b: The upper limit

    Returns:
        The approximate value of the integral.
    """
    h = (b - a) / n
    x = a
    total_sum = f(x)

    for i in range(1, n):
        x = x + h
        total_sum += 2 * f(x)

    total_sum += f(b)
    return (b - a) * total_sum / (2 * n)


def simpsons_rule(f, a, b, n):
    """
    Perform Simpson's 1/3 rule

    Parameters:
        f: The function to integrate.
        n: The number of intervals (must be even)
        a: The lower limit
        b: The upper limit

    Returns:
        The approximate value of the integral.
    """
    if n % 2 != 0:
        raise ValueError("Must be even")

    h = (b - a) / n
    x = a
    total_sum = f(x)

    for i in range(1, n - 1, 2):
        x += h
        total_sum += 4 * f(x)
        x += h
        total_sum += 2 * f(x)

    x += h
    total_sum += 4 * f(x)
    total_sum += f(b)
    return (b - a) * total_sum / (3 * n)


trap_result = trapezoidal_rule(f, 0, 3, 4)
true_error = abs (true_value - trap_result) / true_value * 100

print(f"Trapezoidal result is {trap_result}")
print(f"True Error is: {true_error} %")

trap_result = trapezoidal_rule(f, 0, 3, 100)
print("Trapezoidal rule result:", trap_result)

simpsons_result = simpsons_rule(f, 0, 3, 100)  # Example with 100 intervals
print("Simpson's rule result:", simpsons_result)


x = 0
print(math.e ** x + x)


