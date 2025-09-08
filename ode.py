import numpy as np
import math
import matplotlib.pyplot as plt



def derivative(y, t):
    return y * t ** 2 -1.1 * y

def analytical(t):
    return math.e ** (t ** 3 / 3 - 1.1 * t)

def euler(derivative, t_start, t_end, y0, h):
    t_values = np.arange(t_start, t_end + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    
    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        y = y_values[i - 1]
        y_values[i] = y + h * derivative(y, t)
        
    return t_values, y_values

def mid_point(derivative, t_start, t_end, y0, h):
    t_values = np.arange(t_start, t_end + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    
    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        y = y_values[i - 1]
        
        # Compute the midpoint
        k1 = h * derivative(y, t)
        k2 = h * derivative(y + k1 / 2, t + h / 2)
        
        y_values[i] = y + k2
    
    return t_values, y_values

def runge_kutta_4th(derivative, t_start, t_end, y0, h):
    t_values = np.arange(t_start, t_end + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    
    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        y = y_values[i - 1]
        
        # Compute RK4 slopes
        k1 = h * derivative(y, t)
        k2 = h * derivative(y + k1 / 2, t + h / 2)
        k3 = h * derivative(y + k2 / 2, t + h / 2)
        k4 = h * derivative(y + k3, t + h)
               
        y_values[i] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
               
    return t_values, y_values


y0 = 1
t_start = 0
t_end = 2

t_euler_1, euler_results_1 = euler(derivative, t_start, t_end, y0, 0.5)
print(euler_results_1)

t_euler_2, euler_results_2 = euler(derivative, t_start, t_end, y0, 0.25)
print(euler_results_2)

t_mid, mid_point_results = mid_point(derivative, t_start, t_end, y0, 0.5)
print(mid_point_results)

t_rk4, runge_kutta_4th_results = runge_kutta_4th(derivative, t_start, t_end, y0, 0.5)
print(runge_kutta_4th_results)

t_analytical = np.linspace(t_start, t_end, 1000)
analytical_results = [analytical(t) for t in t_analytical]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_analytical, analytical_results, label="Analytical Solution", linewidth=2, color="black")
plt.plot(t_euler_1, euler_results_1, 'o-', label="Euler (h=0.5)")
plt.plot(t_euler_2, euler_results_2, 'o-', label="Euler (h=0.25)")
plt.plot(t_mid, mid_point_results, 'o-', label="Midpoint (h=0.5)")
plt.plot(t_rk4, runge_kutta_4th_results, 'o-', label="Runge-Kutta 4th (h=0.5)")

# Configure the plot
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.title("Numerical Methods")
plt.legend()
plt.grid()
plt.show()