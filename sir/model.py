import matplotlib.pyplot as plt
import random

def sir(initial, a, b):
    [S, I, R] = initial
    dS = - a * S * I
    dI = a * S * I - b * I
    dR = b * I
    return [S + dS, I + dI, R + dR]
 

def plot_sir(initial, a, b, days, title="SIR Model Simulation", file_path=""):
    S, I, R = initial
    S_list, I_list, R_list = [S], [I], [R]

    # Simulate day by day
    for _ in range(days):
        S, I, R = sir([S, I, R], a, b)
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(S_list, label="Susceptible", linewidth=2)
    plt.plot(I_list, label="Infected", linewidth=2)
    plt.plot(R_list, label="Recovered", linewidth=2)
    plt.xlabel("Days")
    plt.ylabel("Population")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    max_infected = max(I_list)
    return max_infected
 
S0 = 44358
I0 = 3
R0 = 0
number_days = 100

plot_sir([S0, I0, R0], a=1e-10, b=0.025, days=number_days, title="SIR Model for zero infection rate", file_path='sir/zero_infection.png')
plot_sir([S0, I0, R0], a=0.00001, b=1e-10, days=number_days, title="SIR Model for zero recovery rate", file_path='sir/zero_recovery.png')
plot_sir([S0, I0, R0], a=0.00001, b=0.025, days=number_days, title="SIR Model for given parameters", file_path='sir/given_parameters.png')


def run_sir(initial, a, b, days):
    S, I, R = initial
    I_list = [I]

    for _ in range(days):
        S, I, R = sir([S, I, R], a, b)
        I_list.append(I)
    return I_list

def plot_a_sir(initial, a0, b, days, number_runs=100, title="Infection vs Time for infection rate falls into normal distribution", save_path="sir/infection_normal_distribution"):
    """
    Run multiple SIR simulations with a sampled infection rate a ~ N(a0, 0.2*a0)
    and plot I(t) curves.

    Parameters:
        initial (list): Initial values [S, I, R].
        a0 (float): Mean infection rate.
        b (float): Recovery rate.
        days (int): Number of days to simulate.
        number_runs (int): Number of runs to simulate.
        title (str): Plot title.
        save_path (str): Optional file path to save plot.
    """
    plt.figure(figsize=(10, 6))

    # Run for multiple a values
    for _ in range(number_runs):
        a = random.normalvariate(a0, a0 * 0.2)
        I_list = run_sir(initial, a, b, days)
        plt.plot(range(days+1), I_list, color="black", alpha=0.3, linewidth=1)

    # Plot the mean a in red
    I_mean = run_sir(initial, a0, b, days)
    plt.plot(range(days+1), I_mean, color="red", linewidth=2, label=f"a0 = {a0:.2e} (mean)")

    plt.xlabel("Days")
    plt.ylabel("Infected")
    plt.title(title)
    plt.grid()
    plt.legend()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_b_sir(initial, a, b0, days, number_runs=100, title="Infection vs Time for recovery rate falls into normal distribution", save_path="sir/recovery_normal_distribution"):
    """
    Run multiple SIR simulations with a sampled recovery rate b ~ N(b0, b0/6)
    and plot I(t) curves.

    Parameters:
        initial (list): Initial values [S, I, R].
        a (float): Infection rate.
        b0 (float): Mean recovery rate.
        days (int): Number of days to simulate.
        number_runs (int): Number of runs to simulate.
        title (str): Plot title.
        save_path (str): Optional file path to save plot.
    """
    plt.figure(figsize=(10, 6))

    # Run for multiple a values
    for _ in range(number_runs):
        b = random.normalvariate(b0, b0 / 6)
        I_list = run_sir(initial, a, b, days)
        plt.plot(range(days+1), I_list, color="black", alpha=0.3, linewidth=1)

    # Plot the mean a in red
    I_mean = run_sir(initial, a0, b, days)
    plt.plot(range(days+1), I_mean, color="red", linewidth=2, label=f"b0 = {b0:.2e} (mean)")

    plt.xlabel("Days")
    plt.ylabel("Infected")
    plt.title(title)
    plt.grid()
    plt.legend()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


number_runs = 100
a0 = 1e-5
b0 = 0.025

plot_a_sir([S0, I0, R0], a0, b0, number_days, number_runs=number_runs)
plot_b_sir([S0, I0, R0], a0, b0, number_days, number_runs=number_runs)

def plot_peak_infections_histogram(
    initial,
    a0,
    b0,
    days,
    number_runs=100,
    bins=20,
    title="Histogram of Peak Infections",
    save_path="sir/histogram",
):
    """
    For each run:
      - sample a ~ N(a_mean, 0.2*a_mean), b ~ N(b_mean, b_mean/6)
      - simulate SIR for 'days'
      - record peak infected I
    Then plot a histogram (frequency counts) of peak I across runs.

    Returns:
        List[float]: the list of peak infection counts from all runs.
    """
    peaks = []

    for _ in range(number_runs):
        a = random.normalvariate(a0, 0.2 * a0)
        b = random.normalvariate(b0, b0 / 6)
        I_series = run_sir(initial, a, b, days)
        peaks.append(max(I_series))

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(peaks, bins=bins, edgecolor="black")
    plt.xlabel("Peak Infected")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return peaks

plot_peak_infections_histogram([S0, I0, R0], a0, b0, number_days)