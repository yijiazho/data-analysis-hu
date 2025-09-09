import numpy as np
from numpy.polynomial.polynomial import polyfit as npp_polyfit, Polynomial
from scipy.interpolate import KroghInterpolator, BarycentricInterpolator

import matplotlib.pyplot as plt


# ---------------- Utilities ----------------
def divided_differences(x, y):
    """
    Build the divided-difference table dd for nodes x and values y.
    Returns:
      a  : Newton coefficients a[j] = dd[0, j]
      dd : full upper-triangular divided-difference table
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if y.size != n or np.unique(x).size != n:
        raise ValueError("x, y must be same length with all x distinct.")
    dd = np.zeros((n, n), dtype=float)
    dd[:, 0] = y
    for j in range(1, n):
        dd[:n-j, j] = (dd[1:n-j+1, j-1] - dd[:n-j, j-1]) / (x[j:] - x[:n-j])
    a = dd[0, :].copy()
    return a, dd

def newton_to_monomial(a, x_nodes, descending=True):
    """
    Convert Newton-form coefficients 'a' with nodes 'x_nodes' into monomial coefficients.
    Returns ascending powers by default; set descending=True for np.polyval order.
    """
    a = np.asarray(a, dtype=float)
    x_nodes = np.asarray(x_nodes, dtype=float)
    coeffs = np.zeros(1, dtype=float)      # ascending powers
    basis  = np.array([1.0], dtype=float)  # Π_{j<k}(x - x_j)
    for k, ak in enumerate(a):
        # coeffs += ak * basis
        if coeffs.size < basis.size:
            coeffs = np.pad(coeffs, (0, basis.size - coeffs.size))
        coeffs[:basis.size] += ak * basis
        # basis *= (x - x_k)  -> ascending poly for (x - x_k) is [-x_k, 1]
        basis = np.convolve(basis, np.array([-x_nodes[k], 1.0]))
    return coeffs[::-1] if descending else coeffs

# ---------------- Methods that "do the fitting" ----------------
def fit_numpy_polyfit(x, y):
    """
    Uses numpy.polyfit to get monomial coefficients in DESCENDING powers.
    Returns (coeffs_desc, poly1d).
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    deg = len(x) - 1
    coeffs_desc = np.polyfit(x, y, deg=deg)
    p1d = np.poly1d(coeffs_desc)
    return coeffs_desc, p1d

def fit_numpy_polynomial(x, y):
    """
    Uses numpy.polynomial.polynomial.polyfit to get ASCENDING-power coefficients,
    and a Polynomial object.
    Returns (coeffs_asc, Polynomial_object).
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    deg = len(x) - 1
    coeffs_asc = npp_polyfit(x, y, deg)      # ascending powers
    P = Polynomial(coeffs_asc)               # P(x) works directly
    return coeffs_asc, P

def fit_scipy_krogh(x, y):
    """
    SciPy's Newton-form interpolator. Great for interpolation & differentiation.
    Returns the KroghInterpolator object.
    (If you want monomial coefficients, compute divided differences and convert.)
    """
    return KroghInterpolator(x, y)

def fit_scipy_barycentric(x, y):
    """
    SciPy's barycentric Lagrange interpolator. Very stable for evaluating the
    unique degree-(n-1) interpolant. Does not expose monomial coefficients.
    """
    return BarycentricInterpolator(x, y)

# ---------------- Example demo ----------------
if __name__ == "__main__":
    x = np.linspace(1790, 2000, num=22) # every 10 years
    y = np.array([3929000, 5308000, 7240000, 9638000, 12866000, 17069000, 23192000, 31443000, 38558000, 50156000, 62948000, 75995000, 91972000, 105711000, 122755000, 131669000, 150697000, 179323000, 203212000, 226505000, 248709873, 281416000]) 


    # # 1) NumPy: np.polyfit (descending coefficients)
    # coefficient, P = fit_numpy_polyfit(x, y)

    # 2) NumPy: polyseries (ascending coefficients + Polynomial object)
    coefficient, P = fit_numpy_polynomial(x, y)

    # # 3) SciPy: Krogh (Newton form evaluator)
    # krogh = fit_scipy_krogh(x, y)

    # # 4) SciPy: Barycentric (Lagrange form evaluator)
    # bary = fit_scipy_barycentric(x, y)

    # # ---- Divided differences (same for all methods; depends only on x,y) ----
    # a_newton, dd_table = divided_differences(x, y)

    # # If you want monomial coefficients from the Newton form:
    # c_from_newton_desc = newton_to_monomial(a_newton, x, descending=True)

    # Quick checks — all evaluators should agree to numerical precision:
    xx = np.linspace(x.min(), x.max(), 400)
    yy = P(xx)
    # y_np   = p1d(xx)
    # y_np2  = P(xx)            # Polynomial object
    # y_k    = krogh(xx)
    # y_bary = bary(xx)

    # print("max|np.polyfit - Polynomial|   =", np.max(np.abs(y_np - y_np2)))
    # print("max|np.polyfit - Krogh|        =", np.max(np.abs(y_np - y_k)))
    # print("max|np.polyfit - Barycentric|  =", np.max(np.abs(y_np - y_bary)))
    # print("max|np.polyfit - Newton->mono| =", np.max(np.abs(y_np - np.polyval(c_from_newton_desc, xx))))

    # # Divided-difference outputs:
    # print("Newton coefficients a (dd[0, :]):")
    # print(a_newton)
    # # dd_table is upper-triangular; uncomment to inspect:
    # # print(dd_table)


    xx = np.linspace(1780, 2020, 500)
    yy = P(xx)

    plt.figure()
    plt.plot(xx, yy, label="Polyfit")   # smooth curve
    plt.scatter(x, y, marker='o', s=60, label="data")   # original samples
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("polyfit.png", dpi=300)
    plt.show()

    print(P(1968))
    print(P(1999))
    print(P(2020))     
    print(coefficient)