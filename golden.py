import math
from typing import Callable, List
import numpy as np
import matplotlib.pyplot as plt

def golden_section_search_max(func, left, right, max_iterations, min_error, x_values: list[float]=None, rel_errors: list[float]=None):
    """
    Recursively finds the maximum value of a function using Golden Section Search and tracks values for visualization.

    Parameters:
        func (callable): The function to maximize. Should take a float as input and return a float.
        left (float): The left boundary of the search interval.
        right (float): The right boundary of the search interval.
        max_iterations (int): The maximum number of iterations allowed.
        min_error (float): The minimum error rate to stop the recursion.
        x_values (list): A list to store the x values of the midpoint of the intervals.
        rel_errors (list): A list to store the relative errors at each step.

    Returns:
        tuple: A tuple containing:
            - (x_max, f_max): The x and function value at the maximum point.
            - x_values: A list of x values (midpoints of intervals).
            - rel_errors: A list of relative errors at each step.
    """
    if x_values is None:
        x_values = []
    if rel_errors is None:
        rel_errors = []

    # Golden ratio
    golden_ratio = (math.sqrt(5) - 1) / 2

    # Calculate interior points
    c = right - golden_ratio * (right - left)
    d = left + golden_ratio * (right - left)

    # Evaluate the function at interior points
    fc = func(c)
    fd = func(d)

    # Calculate midpoint and relative error
    x_mid = (left + right) / 2
    rel_error = abs(right - left) / abs(x_mid)
    x_values.append(x_mid)
    rel_errors.append(rel_error)

    # Base case: stop if max_iterations reached or interval is small enough
    if max_iterations == 0 or abs(right - left) < min_error:
        x_max = x_mid
        f_max = func(x_max)
        return (x_max, f_max), x_values, rel_errors

    # Recursive step: update the interval
    if fc > fd:
        # The maximum is in the left section
        return golden_section_search_max(func, left, d, max_iterations - 1, min_error, x_values, rel_errors)
    else:
        # The maximum is in the right section
        return golden_section_search_max(func, c, right, max_iterations - 1, min_error, x_values, rel_errors)

R = (math.sqrt(5) - 1 ) / 2

def f(x: float) -> float:
    return -1.5 * x ** 6 - 2 * x ** 4 + 12 * x

def golden_ratio_search(f:Callable[[float], float], left:float, right:float, iter:int=0, max_iterations:int=100, min_error:float=1e-3, x_values:list[float]=None, rel_errors:list[float]=None)-> list[list[float]]:
    """
    Perform Golden Section Search to find the maximum of a function within a specified interval.

    Parameters:
        f (Callable[[float], float]): The function to maximize.
        left (float): The left boundary of the interval.
        right (float): The right boundary of the interval.
        iter (int): The current iteration number, start at 0.
        max_iterations (int): The maximum number of iterations.
        min_error (float): The minimum relative error, in %.
        x_values (list[float]): List to store x values
        rel_errors (list[float]): List to store relative errors
        
    Returns:
        list[list[float]]: A list containing two lists:
            - x_values: The x values
            - rel_errors: The relative errors
    """
    
    #initialize
    if iter == 0:
        x_values = []
        rel_errors = []
        
    # early stopping
    if iter > max_iterations:
        return [x_values, rel_errors]

    # candidates
    x1 = right - R * (right - left)
    x2 = left + R * (right - left)
    f1 = f(x1)
    f2 = f(x2)
    
    # eliminations
    if f1 > f2:
        x_values.append(x1)
        high = x2
        low = left
    else:
        x_values.append(x2)
        low = x1
        high = right
    
    # calculate errors    
    if  iter != 0:
        rel_error = 100 * (1 - R) * abs(right - left) / abs(x_values[-1])
        rel_errors.append(rel_error)
    else:
        rel_errors.append(100)

    # early stopping
    if rel_errors[-1] < min_error:
        return [x_values, rel_errors]
    
    # recursion call
    return golden_ratio_search(f, left=low, right=high, iter=iter+1, max_iterations=max_iterations, min_error=min_error, x_values=x_values, rel_errors=rel_errors)
    
def plot(f: Callable[[float], float], left: float, right: float) -> None:
    x = np.linspace(left, right, 200)
    y = f(x)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.title('Plot of f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.show()
    

def parabolic_interpolation_search(
    f: Callable[[float], float],
    guesses: list[float],
    iter: int = 0,
    max_iterations: int = 100,
    min_error: float = 1e-3,
    x_values: list[float] = None,
    rel_errors: list[float] = None
) -> list[list[float]]:
    """
    Perform Parabolic Interpolation to find the maximum of a function.

    Parameters:
        f (Callable[[float], float]): The function to maximize.
        guesses (List[float]): A list of three initial guesses [x0, x1, x2].
        iter (int): The current iteration number
        max_iterations (int): Maximum number of iterations allowed
        min_error (float): Minimum relative error for stopping the search in %
        x_values (List[float]): List to store x values during iterations
        rel_errors (List[float]): List to store relative errors during iterations

    Returns:
        list[list[float]]: A list containing two lists:
            - x_values: The x values evaluated during the search.
            - rel_errors: The relative errors computed at each iteration.
    """
    # Initialize x_values and rel_errors on the first iteration
    if iter == 0:
        x_values = []
        rel_errors = []

    # termination
    if iter > max_iterations:
        return [x_values, rel_errors]
    
    x0, x1, x2 = guesses
    f0, f1, f2 = f(x0), f(x1), f(x2)

    # Parabolic interpolation formula
    A = f0 * (x1 ** 2 - x2 ** 2) + f1 * (x2 ** 2 - x0 ** 2) + f2 * (x0 ** 2 - x1 ** 2)
    B = 2 * (f0 * (x1 - x2) + f1 * (x2 - x0) + f2 * (x0 - x1))
    if abs(B) < 1e-6:
        return [x_values, rel_errors]
    x_new = A / B

    # Compute relative error
    if iter != 0:
        rel_error = 100 * abs(x_new - x_values[-1]) / abs(x_values[-1])
        rel_errors.append(rel_error)
    else:
        rel_errors.append(100)

    x_values.append(x_new)

    # Terminate if relative error is below the threshold
    if rel_errors[-1] < min_error:
        return [x_values, rel_errors]

    # Terminate if maximum iterations are exceeded
    if iter >= max_iterations:
        return [x_values, rel_errors]

    # Update guesses for the next iteration
    # Replace the lowest because we want to find max
    if f0 < f1 and f0 < f2:
        next_guesses = [x_new, x1, x2]
    elif f1 < f0 and f1 < f2:
        next_guesses = [x0, x_new, x2]
    else:
        next_guesses = [x0, x1, x_new]

    return parabolic_interpolation_search(
        f, next_guesses, iter=iter + 1, max_iterations=max_iterations,
        min_error=min_error, x_values=x_values, rel_errors=rel_errors
    )   
    
    
# Example usage
if __name__ == "__main__":
    xs, errors = golden_ratio_search(f, 0, 1.5)
    print(f'The max value of function is {f(xs[-1])}, and the coressponding x is {xs[-1]}')    
    
    plt.figure(figsize=(8, 6))
    plt.plot(errors)
    plt.xlabel('Iterations')
    plt.ylabel('Relative Errors(%)')
    plt.show()
    
    
    xss, errorss = parabolic_interpolation_search(f, [0, 1, 2], max_iterations=3)
    print(f'The max value of function is {f(xss[-1])}, and the coressponding x is {xss[-1]}')
    
    plt.figure(figsize=(8, 6))
    plt.plot(errorss)
    plt.xlabel('Iterations')
    plt.ylabel('Relative Errors(%)')
    plt.show()
    
    
