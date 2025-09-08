
import cmath
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f1(x: float) -> float:
    return x ** 3 - x ** 2 + 2 * x - 2

def f2(x: float) -> float:
    return 2 * x ** 4 + 6 * x ** 2 + 8

def f3(x: float) -> float:
    return -2 + 6.2 * x - 4 * x ** 2 + 0.7 * x ** 3

def f4(x: float) -> float:
    return x ** 4 - 2 * x ** 3 + 6 * x ** 2 - 2 * x + 5


def muller(f, x0: float, x1: float, x2: float, iteration: int) -> list[dict]:
    results = []
    
    for i in range(iteration):
        h0 = x1 - x0
        h1 = x2 - x1
        delta0 = (f(x1) - f(x0)) / h0
        delta1 = (f(x2) - f(x1)) / h1
        a = (delta1 - delta0) / (h1 + h0)
        b = a * h1 + delta1
        c = f(x2)
        
        # calculate the discriminant
        discriminant = cmath.sqrt(b * b - 4 * a * c)
        
        # two possible roots, choose the one with the larger magnitude of the denominator
        if abs(b + discriminant) > abs(b - discriminant):
            root = x2 + (-2 * c) / (b + discriminant)
        else:
            root = x2 + (-2 * c) / (b - discriminant)
        
        if i > 0:
            error = abs(root - lastRoot)
            relative_error = error / root
        else:
            error = None
            relative_error = 1
        
        
        is_real = root.imag == 0
        
        results.append({
            "iteration": i+1,
            "root": root,
            "is_real": is_real,
            "error": error,
            "relative_error": relative_error
        })
        
        x0 = x1
        x1 = x2
        x2 = root
        lastRoot = root
    
    return results
    
def plot3D(f, real_left=-2, real_right=2, imaginary_left=-2, imaginary_right=2)-> None:
    
    # Create a meshgrid for complex inputs
    real_vals = np.linspace(real_left, real_right, 400)
    imag_vals = np.linspace(imaginary_left, imaginary_right, 400)
    X, Y = np.meshgrid(real_vals, imag_vals)
    Z = X + 1j * Y
    W = f(Z)

    # Calculate the magnitude
    magnitude = np.abs(W)

    # Create a figure with subplots for different viewing angles
    fig = plt.figure(figsize=(16, 12))

    # Define viewing angles (elevation and azimuth)
    angles = [(30, 30), (30, 120), (30, 210), (30, 300)]

    # Generate subplots for each angle
    for i, angle in enumerate(angles):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        
        # Plot the surface
        ax.plot_surface(X, Y, magnitude, cmap='viridis', alpha=1.0)
        
        # Plot the contour where |f1(x)| is close to 0 (indicating roots)
        epsilon = 1e-6
        ax.contour(X, Y, magnitude, levels=[epsilon], colors='r', linestyles='--')
        
        # Set viewing angle
        ax.view_init(elev=angle[0], azim=angle[1])
        
        # Labels and title
        ax.set_xlabel('Real(X)')
        ax.set_ylabel('Imaginary(Y)')
        ax.set_zlabel('|f1(X + iY)|')
        ax.set_title(f'View Angle (Elev: {angle[0]}, Azim: {angle[1]})')

    plt.tight_layout()
    plt.show()


def bairstow(coeffs, r: complex, s: complex, iteration: int) -> list[dict]:
    eps = 1e-6
    n = len(coeffs) - 1  # Degree of the polynomial
    roots = []

    def divide(a, r, s):
        b = np.zeros(len(a), dtype=complex)
        c = np.zeros(len(a), dtype=complex)
        
        b[n] = a[n]
        b[n - 1] = a[n - 1] + r * b[n]
        
        for i in range(n - 2, -1, -1):
            b[i] = a[i] + r * b[i + 1] + s * b[i + 2]
        
        c[n] = b[n]
        c[n - 1] = b[n - 1] + r * c[n]
        
        for i in range(n - 2, -1, -1):
            c[i] = b[i] + r * c[i + 1] + s * c[i + 2]
        
        return b, c


    for iteration in range(iteration):
        b, c = divide(coeffs, r, s)
        
        # Compute the deltas for r and s
        det = c[2]*c[2] - c[3]*c[1]
        if abs(det) < eps:
            break
        
        dr = (-b[1]*c[2] + b[0]*c[3]) / det
        ds = (-b[0]*c[2] + b[1]*c[1]) / det
        
        r += dr
        s += ds
    
    # After finding r and s, determine the roots from the quadratic factor
    discriminant = r ** 2 + 4 * s
    
    root1 = (r + np.sqrt(discriminant)) / 2
    root2 = (r - np.sqrt(discriminant)) / 2
    
    roots.append(root1)
    roots.append(root2)
    
    b, _ = divide(coeffs, r, s)
    
    if n - 2 > 2:
        # Apply Bairstow's method recursively by removing top 2 elements
        roots.extend(bairstow(b[2:], r, s, iteration))
    elif n - 2 == 2:
        # Solve the remaining quadratic polynomial
        A, B, C = b[-1], b[-2], b[-3]
        discriminant = B ** 2 - 4 * A * C
        root1 = (-B + np.sqrt(discriminant)) / (2 * A)
        root2 = (-B - np.sqrt(discriminant)) / (2 * A)
        
        roots.append(root1)
        roots.append(root2)

    elif n - 2 == 1:
        # Solve the remaining linear polynomial
        roots.append(-s / r)

    return roots
    
def g(x):
    return 1.25 -3.875 * x + 2.125 * x**2 + 2.75 * x**3 -3.5 * x ** 4 + x ** 5
    
    
def main():
    # plot3D(f3)
    initial_gusses = [1.2, 1.1, 0.9]
    iterations = 5
    
    results_muller_root1 = muller(f1, *initial_gusses, iterations)
    
    for result in results_muller_root1:
        print(
            f"Iteration {result['iteration']}: "
            f"Root = {result['root']}, "
            f"Real = {result['is_real']}, "
            f"Error = {result['error']}, "
            f"Relative Error = {100 * result['relative_error']}%"
        )    
    
    #poly_coeffs = [1.25, -3.875, 2.125, 2.75, -3.5, 1]
    poly_coeffs = [8, 0, 6, 0, 2]
    initial_r = 2.5 + 2.5j
    initial_s = -0.5j
    iterations = 10
    results_bairstow = bairstow(poly_coeffs, initial_r, initial_s, iterations)

    for result in results_bairstow:
        print(f"Root: {result}")
        print(f" f(x): {f2(result)}")
        
    
    
    
if __name__ == "__main__":
    main()