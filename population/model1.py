from math import factorial
import numpy as np
from scipy.interpolate import BarycentricInterpolator, KroghInterpolator, lagrange
from numpy.polynomial import Polynomial, Chebyshev

import matplotlib.pyplot as plt



# --------- Method 2: Newton divided differences ----------
def newton_divided_differences(x, y):
    """
    Return Newton form coefficients a and nodes x such that
    p(t) = a[0] + a[1](t-x0) + a[2](t-x0)(t-x1) + ... (nested form).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    coef = y.astype(float).copy()
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
    return coef  # Newton coefficients a[0..n-1]

def newton_eval(a, x_nodes, t):
    """
    Evaluate Newton-form polynomial with coefficients a and nodes x_nodes at t.
    t can be scalar or array.
    """
    t = np.asarray(t, dtype=float)
    n = len(a)
    p = np.zeros_like(t) + a[-1]
    for k in range(n-2, -1, -1):
        p = a[k] + (t - x_nodes[k]) * p
    return p


def monomial_coeffs_from_interpolator(P, degree, center=None):
    """
    Given a SciPy polynomial interpolator P with .derivatives, return monomial
    coefficients (highest power first) of degree <= degree.
    """
    if center is None:
        # a numerically sensible default center
        try:
            # use the mean of its nodes if exposed
            center = float(np.mean(P.xi))
        except Exception:
            center = 0.0
    ders = np.asarray(P.derivatives(center, der=degree), dtype=float)  # [p, p', ..., p^(degree)]
    a = np.array([ders[k]/factorial(k) for k in range(degree+1)], dtype=float)  # Taylor coeffs at center
    return _taylor_to_monomial(a, center)

def _taylor_to_monomial(a, c):
    """
    Convert Taylor coefficients a[k] around center c (for (x-c)^k) to monomial
    coefficients in increasing power order, then return in decreasing order
    (np.polyval-compatible).
    """
    n = len(a)
    # Work in "low-to-high" order internally
    coeffs_low = np.zeros(n, dtype=float)
    basis = np.array([1.0])              # (x-c)^0
    for k, ak in enumerate(a):
        coeffs_low[:basis.size] += ak * basis
        basis = np.convolve(basis, np.array([-c, 1.0]))  # multiply by (x - c)
    return coeffs_low[::-1]              # return "high-to-low" for np.polyval

# ---------------- Example & quick check ----------------
if __name__ == "__main__":
    x = np.linspace(1790, 2000, num=22) # every 10 years
    y = np.array([3929000, 5308000, 7240000, 9638000, 12866000, 17069000, 23192000, 31443000, 38558000, 50156000, 62948000, 75995000, 91972000, 105711000, 122755000, 131669000, 150697000, 179323000, 203212000, 226505000, 248709873, 281416000]) 

    y_log = np.log1p(y)
    # y = y / 1e6

    # # Newton form
    # a_newton = newton_divided_differences(x, y)
    # print("Newton coefficients:", a_newton)
    # # Evaluate Newton form at x and a dense grid
    # y_check_n = newton_eval(a_newton, x, x)
    # print("Exact match (Newton)?", np.allclose(y_check_n, y))

    # Barycentric (stable, exact interpolation)
    P = BarycentricInterpolator(x, y)
    print(P(0.5))            # evaluate
    print(np.allclose(P(x), y))   # True (exact within fp error)
    P_log = BarycentricInterpolator(x, y_log)
    print(P_log(0.5))            # evaluate
    print(np.allclose(P_log(x), y_log))   # True (exact within fp error)



    # # Krogh (Newton form; also exact)
    # P = KroghInterpolator(x, y)
    # print(P([0.5, 2.0]))

    # # Lagrange (gives monomial coefficients via poly1d; less stable)
    # P = lagrange(x, y)       # numpy.poly1d
    # print(P.c)               # coefficients (highest power first)
    # P_log = lagrange(x, y_log)
    # print(P_log.c)               # coefficients (highest power first)


    # cheb = Chebyshev.fit(x, y, deg=len(x)-1)   # LS fit; with deg = n-1 it typically interpolates
    # P = cheb.convert(kind=Polynomial)     # monomial basis
    # # coefficients in ASCENDING power: P.coef
    # print(np.allclose(P(x), y))   # True (exact within fp error)
    # print(P.coef)

    # np polynomial
    # coeffs = np.polyfit(x, y, deg=len(x)-1)
    # P = np.poly1d(coeffs)
    # print(coeffs)

    V = np.vander(x, N=x.size, increasing=True)
    coeffs = np.linalg.solve(V, y)
    print("coefficients:", coeffs)

    xx = np.linspace(1790, 2000, 500)
    yy = P(xx)

    plt.figure()
    plt.plot(xx, yy, label="Barycentric interpolant")   # smooth curve
    plt.scatter(x, y, marker='o', s=60, label="Data points")   # original samples
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("interpolation_barycentric.png", dpi=300)
    plt.show()
    print(P(1968))
    print(P(1999))
    print(P(2020)) 

