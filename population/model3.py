import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

def divided_differences(x, y):
    """
    Returns the full divided-difference table (n x n).
    Assumes x has unique values. Sorts (x, y) by x for robustness.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # sort by x (divided differences assume ordered abscissae)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if np.unique(x).size != x.size:
        raise ValueError("x must contain unique values for divided differences.")

    n = len(x)
    dd = np.zeros((n, n), dtype=float)
    dd[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            denom = x[i + j] - x[i]
            dd[i, j] = (dd[i + 1, j - 1] - dd[i, j - 1]) / denom

    return x, y, dd

def choose_degree_via_dd(dd, threshold=0.01):
    """
    Choose degree by the first column of order j (j>=1) such that
    ALL entries in that column have |value| < threshold.
    Returns degree = j - 1. If no such column, return full degree n-1.
    """
    n = dd.shape[0]
    best_degree = n - 1
    for j in range(1, n):
        col = dd[: n - j, j]
        if np.all(np.abs(col) < threshold):
            best_degree = j - 1
            break
    return best_degree

def fit_low_order_poly(x, y, threshold=0.01, rcond=None):
    """
    1) Compute divided differences and pick degree.
    2) Least-squares fit of that degree with np.polyfit.
    3) Return (degree, coefficients_high_to_low, Polynomial object).
    """
    x_sorted, y_sorted, dd = divided_differences(x, y)
    deg = choose_degree_via_dd(dd, threshold=threshold)

    # Make sure degree is at least 0 and strictly less than len(x)
    deg = int(max(0, min(deg, len(x_sorted) - 1)))

    # Least-squares fit (robust to noise); returns highest-degree-first coefficients
    coefs_desc = np.polyfit(x_sorted, y_sorted, deg=deg, rcond=rcond)

    # Convert to numpy.polynomial.Polynomial (ascending order)
    poly = Polynomial(coefs_desc[::-1])
    return deg, coefs_desc, poly

def plot_data_and_poly(x, y, poly, num=400, title=None):
    """
    Plot original points and the fitted polynomial curve.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    xs = np.linspace(x_min, x_max, num=num)
    ys = poly(xs)

    plt.figure()
    plt.scatter(x, y, label="Data points")  # default styling
    plt.plot(xs, ys, label="Fitted polynomial")  # default styling
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.title(title or "Low-order polynomial fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig("polynomial_fit.png", dpi=300)
    plt.show()

# ---------------------------
# Example usage / demo
# ---------------------------
if __name__ == "__main__":
    # If you already have x, y as numpy arrays, replace this demo block with your data.
    x = np.linspace(1790, 2000, num=22) # every 10 years
    y = np.array([3929000, 5308000, 7240000, 9638000, 12866000, 17069000, 23192000, 31443000, 38558000, 50156000, 62948000, 75995000, 91972000, 105711000, 122755000, 131669000, 150697000, 179323000, 203212000, 226505000, 248709873, 281416000]) 


    threshold = 0.01  # you can tweak this
    deg, coefs_desc, P = fit_low_order_poly(x, y, threshold=threshold)

    print(f"Chosen degree: {deg}")
    print("Polynomial coefficients (highest degree first, np.polyfit convention):")
    print(coefs_desc)

    # Also show ascending-order coefficients (Polynomial convention)
    print("\nPolynomial (ascending order) coefficients (constant -> x^deg):")
    print(P.coef)

    # Nicely formatted polynomial
    terms = " + ".join(
        f"{coef:.6g}·x^{i}" if i > 1 else ("{:.6g}·x".format(coef) if i == 1 else "{:.6g}".format(coef))
        for i, coef in enumerate(P.coef)
    )
    print(f"\nP(x) = {terms}")
    print(P(1968))
    print(P(1999))
    print(P(2020))   

    # Plot
    plot_data_and_poly(x, y, P, title=f"Best-fit polynomial (degree {deg})")
