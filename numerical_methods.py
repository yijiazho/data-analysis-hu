import numpy as np
import matplotlib.pyplot as plt
import math

## real roots of f(x) = 4x^3 - 6x^2 + 7x -2.3

# def f(x: float) -> float:
#     return x ** 5 - 8 * x ** 4 + 44 * x ** 3 - 91 * x ** 2 + 85 * x - 26


def bisect(low: float, high: float) -> list[list[float]]:
    roots = []
    errors = []
    error = 1
    lastRoot = low
    
    while error > 0.1:
        mid = (low + high) / 2
        y = f(mid)
        if y > 0:
            high = mid
        else: 
            low = mid
        error = abs((mid - lastRoot) / mid)
        roots.append(mid)
        errors.append(error)
        
        print(f'Guess value of root: {mid}')
        print(f'Relative Error: {100 * error} %')
        lastRoot = mid
    return [roots, errors]
        
def falsepositive(low: float, high: float) -> list[list[float]]:
    roots = []
    errors = []
    error = 1
    lastRoot = low
    
    while error > 0.002:
        f_low = f(low)
        f_high = f(high)
        root = (high * f_low - low * f_high) / (f_low - f_high)
        if f(root) > 0:
            high = root
        else:
            low = root
        error = abs((root - lastRoot) / root)
        
        roots.append(root)
        errors.append(error)
        print(f'Guess value of root: {root}')
        print(f'Relative Error: {100 * error} %')
                
        lastRoot = root
    return [roots, errors]

def g(x: float) -> float:
    return (6 + 11.6 * x**2 - 2.1 * x**3) / 17.5

def f(x: float) -> float:
    return 2.1 * x ** 3 - 11.6 * x ** 2 + 17.5 * x - 6

def f_prime(x: float) -> float:
    return 6.3 * x ** 2 - 23.2 * x + 17.5

def fixpoint(x: float, iteration: int) -> list[list[float]]:
    roots = []
    errors = []
    lastRoot = x
    error = 1
    
    for i in range(iteration):
        root = g(lastRoot)
        error = abs((root - lastRoot) / root)
        
        roots.append(root)
        errors.append(error) 
        
        lastRoot = root
    
    return [roots, errors]
        
def newton_raphson(x: float, iteration: int) -> list[list[float]]:
    roots = []
    errors = []
    lastRoot = x
    error = 1
    
    for i in range(iteration):
        root = lastRoot - f(lastRoot) / f_prime(lastRoot)
        error = abs((root - lastRoot) / root)
        
        roots.append(root)
        errors.append(error) 
        
        lastRoot = root
    
    return [roots, errors]   

def secant(prev: float, x: float, iteration: int) -> list[list[float]]:
    roots = []
    errors = []
    lastRoot = x
    lastLastRoot = prev
    error = 1
    
    for i in range(iteration):
        root = lastRoot - (f(lastRoot) * (lastLastRoot - lastRoot)) / (f(lastLastRoot) - f(lastRoot))
        error = abs((root - lastRoot) / root)
        
        roots.append(root)
        errors.append(error) 
        
        lastLastRoot = lastRoot
        lastRoot = root
    
    return [roots, errors] 


def modified_secant(x: float, delta: float, iteration: int) -> list[list[float]]:
    roots = []
    errors = []
    lastRoot = x
    error = 1
    
    for i in range(iteration):
        root = lastRoot - (delta * lastRoot * f(lastRoot)) / (f(lastRoot + delta * lastRoot) - f(lastRoot))
        error = abs((root - lastRoot) / root)
        
        roots.append(root)
        errors.append(error) 
        
        lastRoot = root
    
    return [roots, errors] 

def main():
    bisect_roots, bisect_errors = bisect(0.5, 1)
    falsepositive_roots, falsepositive_errors = falsepositive(0.5, 1)
    
    # plt.plot(bisect_roots, label='Bisection Method')
    # plt.plot(falsepositive_roots, label='False Position Method')
    # plt.xlabel('Iteration')
    # plt.ylabel('Guess value of root')
    # plt.title('Root Guess Values Over Iterations')
    # plt.legend()
    # plt.grid(True)
    
    # bisect_errors.insert(0, 1)
    # falsepositive_errors.insert(0, 1)
    # plt.plot(bisect_errors, label='Bisection Method')
    # plt.plot(falsepositive_errors, label='False Position Method')
    # plt.xlabel('Iteration')
    # plt.ylabel('Relative Error (log)')
    # plt.title('Relative Error Over Iterations')
    # plt.yscale('log')
    # plt.legend()
    
    
    # plt.show()
    
    # x = np.linspace(3, 4, 100)
    # dx = x[1]-x[0]
    # y = g(x)
    # dydx = np.gradient(y, dx)
    
    # plt.plot(x, dydx, label="dy/dx")
    # plt.xlabel("x")
    # plt.ylabel("dy/dx")
    # plt.title("Plot of dy/dx vs. x")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    
    fixpoint_roots, fixpoint_errors = fixpoint(3, 3)
    for i in range(3):
        print(f'Guess value of root: {fixpoint_roots[i]}')
        print(f'Relative error: {100 * fixpoint_errors[i]}%')
        
    print('-------------------------------------------------')
    
    newton_raphson_roots, newton_raphson_errors = newton_raphson(3, 3)
    for i in range(3):
        print(f'Guess value of root: {newton_raphson_roots[i]}')
        print(f'Relative error: {100 * newton_raphson_errors[i]}%')

    print('-------------------------------------------------')
    
    secant_roots, secant_errors = secant(3, 4, 3)
    for i in range(3):
        print(f'Guess value of root: {secant_roots[i]}')
        print(f'Relative error: {100 * secant_errors[i]}%')

    print('-------------------------------------------------')
    
    modified_secant_roots, modified_secant_errors = modified_secant(3, 0.01, 3)
    for i in range(3):
        print(f'Guess value of root: {modified_secant_roots[i]}')
        print(f'Relative error: {100 * modified_secant_errors[i]}%')  
        
        
    fixpoint_errors.insert(0, 1)
    newton_raphson_errors.insert(0, 1)
    secant_errors.insert(0, 1)
    modified_secant_errors.insert(0, 1)
    
    plt.plot(fixpoint_errors, label='Fix-point Method')
    plt.plot(newton_raphson_errors, label='Newton_Raphson Method')
    plt.plot(secant_errors, label="Secant Method")
    plt.plot(modified_secant_errors, label="Modified Secant Method")
    
    plt.xlabel('Iteration')
    plt.ylabel('Relative Error (log)')
    plt.title('Relative Error Over Iterations')
    plt.yscale('log')
    plt.legend()   
    plt.show()   

if __name__ == "__main__":
    main()