import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 14 days
gamma = 1/14
# recovery rate = R0 * gamma
beta = 0.4
# US population 1/5
N = 66e6
# from single patient 0
I0 = 1
S0 = N - I0
R0 = 0
dt = 5.5
days = 320
steps = int(days / dt)

# find [S, I, R] recursively
def sir(y, beta, gamma):
    S, I, R = y
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return [S + dS, I + dI, R + dR]


S, I, R = S0, I0, R0
S_time = [S]
I_time = [I]
R_time = [R]
time=[0]

for _ in range(steps):
    [S, I, R] = sir([S, I, R], beta, gamma)
    S_time.append(S)
    I_time.append(I)
    R_time.append(R)
    time.append(time[-1] + dt)
    

# load original table, downloaded from kaggle
path = "us_covid19_daily.csv"
df = pd.read_csv(path)

# pre-process the table to only keep date, s, i, r
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df = df.sort_values('date')


row_sums = df[['recovered', 'death']].sum(axis=1)
max_sum = row_sums.max()

max_positive = df['positive'].max()


df['recovered'] = df['recovered'].fillna(0)
df['death'] = df['death'].fillna(0)
df['Recovered'] = df['recovered'] + df['death']
df['positive'] = df['positive'].fillna(0)
df['Infected'] = df['positive'] - df['Recovered']
df['Susceptible'] = N - df['Infected'] - df['Recovered']

sir_df = df[['date', 'Susceptible', 'Infected', 'Recovered']].copy()
sir_df = sir_df.reset_index(drop=True)
sir_df[['Susceptible', 'Infected', 'Recovered']] = sir_df[['Susceptible', 'Infected', 'Recovered']].apply(pd.to_numeric)

print(sir_df.head())

sir_df.to_csv("covid_us_sir.csv")

actual_infected = sir_df['Infected'].values
actual_recovered = sir_df['Recovered'].values
actual_susceptible = sir_df['Susceptible'].values
time = np.arange(len(actual_infected))

time_model = np.arange(0, steps + 1) * dt  # same length as S_time, I_time, etc.

# Interpolate model to match real data's time axis
I_time_interp = np.interp(time, time_model, I_time)
R_time_interp = np.interp(time, time_model, R_time)
S_time_interp = np.interp(time, time_model, S_time)


plt.figure(figsize=(12, 6))
plt.plot(time, actual_infected, label='Actual Infected', color='red')
plt.plot(time, actual_recovered, label='Actual Recovered', color='green')
plt.plot(time, actual_susceptible, label='Actual Susceptible', color='blue')

plt.plot(time, I_time_interp, '--', label='Modeled Infected', color='red', alpha=0.6)
plt.plot(time, R_time_interp, '--', label='Modeled Recovered', color='green', alpha=0.6)
plt.plot(time, S_time_interp, '--', label='Modeled Susceptible', color='blue', alpha=0.6)

plt.xlabel("Days since 2020-01-22")
plt.ylabel("Population")
plt.title("SIR Model vs Actual COVID-19 Data (US)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()