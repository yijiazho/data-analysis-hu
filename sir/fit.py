import numpy as np
from scipy.optimize import minimize
import pandas as pd

# Load data
path = "covid_us_sir.csv"
df = pd.read_csv(path)

# Extract actual data
actual_infected = df['Infected'].values
actual_recovered = df['Recovered'].values
actual_susceptible = df['Susceptible'].values
time = np.arange(len(actual_infected))

# Initial conditions
S0 = actual_susceptible[0]
I0 = actual_infected[0]
R0 = actual_recovered[0]
N = S0 + I0 + R0

print("Initial conditions:", S0, I0, R0, N)

# SIR model step with dt
def sir_step(y, beta, gamma, dt):
    S, I, R = y
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return [S + dS * dt, I + dI * dt, R + dR * dt]

# Objective function: fit beta, gamma, and dt
def objective(params):
    beta, gamma, dt = params
    S, I, R = S0, I0, R0
    modeled_I = [I]
    for _ in range(1, len(actual_infected)):
        S, I, R = sir_step([S, I, R], beta, gamma, dt)
        modeled_I.append(I)
    modeled_I = np.array(modeled_I)
    valid = actual_infected > 0
    return np.mean((actual_infected[valid] - modeled_I[valid])**2)

# Initial guesses: beta, gamma, dt
initial_guess = [0.4, 1/14, 4]
bounds = [(0.01, 1.0), (0.01, 1.0), (0.01, 10.0)]  # reasonable range for dt

# Run optimization
result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

# Extract fitted parameters
fitted_beta, fitted_gamma, fitted_dt = result.x
print(f"Fitted beta: {fitted_beta:.4f}")
print(f"Fitted gamma: {fitted_gamma:.4f}")
print(f"Fitted dt: {fitted_dt:.4f}")
print(f"Estimated R0: {fitted_beta / fitted_gamma:.2f}")
