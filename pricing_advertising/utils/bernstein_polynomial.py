import numpy as np
from scipy.special import comb


# Define the Bernstein polynomial
def bernstein_polynomial(k, M, x):  # noqa: ANN201
    """Calculate the Bernstein polynomial ψk(x, M)."""
    return comb(M, k) * (x**k) * ((1 - x) ** (M - k))


def theta_to_beta(theta: np.ndarray) -> np.ndarray:
    """
    Convert θ coefficients to β using cumulative sum.

    Args:
        theta (numpy.ndarray): θ coefficients.

    Returns:
        numpy.ndarray: β coefficients.
    """
    M = len(theta) - 1
    beta = np.zeros_like(theta)
    for k in range(M + 1):
        beta[k] = np.sum(theta[k:])  # Reverse cumulative sum
    return beta


def pricing_curve(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Calculate f(x) for a monotonically decreasing function using θ coefficients.

    Args:
        x (numpy.ndarray): Array of values in [0, 1].
        theta (numpy.ndarray): θ coefficients (all non-negative).

    Returns:
        numpy.ndarray: f(x) values.
    """
    beta = theta_to_beta(theta)  # Convert θ to β
    M = len(beta) - 1
    f_values = np.zeros_like(x)
    for k in range(M + 1):
        f_values += beta[k] * bernstein_polynomial(k, M, x)
    return f_values
