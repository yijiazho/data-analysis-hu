import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random

random.seed(42)
width = 10
demands = np.array([20,30,40,50,60,70,80,90,100])
bin_centers = np.array([25,35,45,55,65,75,85,95])
days = np.array([208,160,166,112,17,43,13,11])
total_days = np.sum(days)

probs = np.divide(days, total_days)
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
plt.savefig('inventory_model/histogram.png', dpi=300, bbox_inches='tight')
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
plt.plot(x_new, y_new, "-", label="Linear spline submodel")
plt.title("Linear spline model for the demand submodel")
plt.xlabel("Demands (gallon)")
plt.ylabel("Cumulative Frequency")
plt.legend()
plt.savefig('inventory_model/interpolation.png', dpi=300, bbox_inches='tight')
plt.show()


def random_q():
    rand_prob = random.random()
    for demand in reversed(demands):
        if rand_prob > f_linear(demand):
            return demand + width / 2
    return demands[0] + width / 2

def median_q():
    return 35

def average_q():
    weighted_sum = np.sum(days * bin_centers)
    total_days = np.sum(days)
    return weighted_sum / total_days

print("Median demand:", median_q())
print("Average demand:", average_q())

def monte_carlo(Q, T, d, s, N, q_func):
    """
    Run Monte Carlo simulation and return average cost per day.

    Parameters:
        Q (float): delivery quantity
        T (int): time between deliveries
        d (float): delivery cost
        s (float): holding cost per unit per day
        N (int): total number of days
        q_func (function): demand generator (e.g., random_q or median_q)
    """
    K = N # days remaining
    I = 0 # inventory storage
    C = 0 # total cost
    # flag for exiting the loop
    flag = False

    # start of a inventory cycle
    while flag == False:
        I = I + Q
        C = C + d
        if (T >= K):
            T = K
            flag = True
        
        for i in range(1, T + 1):
            q = q_func()
            I = I - q
            if (I < 0):
                I = 0
                K = K - 1
                break
            else:
                C = C + I * s
            K = K - 1


    return C / N
        
total = 0        
for i in range(1000):
    total += monte_carlo(800, 15, 92.0, 0.001, 365, median_q)
print("Average cost based on median demand:", total / 1000)

total = 0
for i in range(1000):
    total += monte_carlo(800, 15, 92.0, 0.001, 365, average_q)
print("Average cost based on average demand:", total / 1000)


def plot_cost_histogram(Q, T, d, s, N, q_func, runs=1000, bins=30):
    """
    Run Monte Carlo simulation multiple times and plot histogram of average daily cost.

    Parameters:
        Q (float): order quantity
        T (int): maximum cycle length
        d (float): fixed order cost
        s (float): holding cost per unit per day
        N (int): horizon (days)
        q_func (function): demand generator (e.g., random_q or median_q)
        runs (int): number of Monte Carlo runs
        bins (int): number of histogram bins
    """
    results = [monte_carlo(Q, T, d, s, N, q_func) for _ in range(runs)]

    plt.hist(results, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel("Average Daily Cost")
    plt.ylabel("Number of Occurrences")
    plt.title(f"Distribution of Average Daily Cost over {runs} runs")
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('inventory_model/cost_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_cost_histogram(800, 15, 92.0, 0.001, 365, random_q, runs=1000)
