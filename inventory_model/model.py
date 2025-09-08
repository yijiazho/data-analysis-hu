import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random


demands = np.array([20,30,40,50,60,70,80,90,100])
days = np.array([208,160,166,112,17,43,13,11])
sumdays = np.sum(days)

probs = np.divide(days, sumdays)
cumdays = np.cumsum(days)
cumprobs = np.cumsum(probs)



plt.bar(demands[:-1], probs, width=np.diff(demands), align='edge',alpha=0.5, edgecolor='k')
for xi, yi, wi in zip(demands[:-1], probs, np.diff(demands)):
    plt.text(xi + wi/2, yi + 0.01, f"{yi:.6g}", 
             ha='center', va='bottom', color='blue')

plt.xlabel("Demands (gallon)")
plt.ylabel("Frequency")
plt.title("Frequency of each demand")
plt.legend()
plt.show()


cumprobs = np.insert(cumprobs, 0, 0.0)
print(cumprobs)


# Create a linear interpolator
f_linear = interp1d(demands, cumprobs, kind='linear')

# New x values (more fine-grained)
x_new = np.linspace(demands.min(), demands.max(), 200)
y_new = f_linear(x_new)

# Plot original points and interpolated line
plt.plot(demands, cumprobs, "o", label="Data points")
plt.plot(x_new, y_new, "-", label="Linear interpolation")
plt.legend()
plt.show()


def random_q():
    rand_prob = random.random()
    for demand in reversed(demands):
        if rand_prob > f_linear(demand):
            return 
        


def monte_carlo(Q, T, d, s, N):
    K = N
    I = 0
    C = 0
    # flag for exiting the loop
    flag = false

    I = I + Q
    C = C + d
    if (T >= K):
        T = K
        flag = true
    
    for i in 1 to T + 1:
        x = f_linear(Q)
        
    