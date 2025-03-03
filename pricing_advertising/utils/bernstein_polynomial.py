import numpy as np
from scipy.special import comb


# Define the Bernstein kth-basis polynomial
def bernstein_basis_polynomial(k: int, M: int, x: np.ndarray | float) -> np.ndarray | float:
    """Calculate the Bernstein kth-basis polynomial ψk(x, M)."""
    return comb(M, k) * (x**k) * ((1 - x) ** (M - k))


def theta_to_beta(theta: np.ndarray) -> np.ndarray:
    """
    Convert θ coefficients to β using cumulative sum.

    Args:
        theta (numpy.ndarray): θ coefficients.

    Returns:
        numpy.ndarray: β coefficients.
    """
    # theta_M = beta_M
    # theta_k =  beta_k - beta_{k+1}
    M = len(theta) - 1
    beta = np.zeros_like(theta)
    for k in range(M + 1):
        beta[k] = np.sum(theta[k:])  # Reverse cumulative sum
    return beta


def bernstein_monotonic_decreasing_polynomial(
    x: np.ndarray | float, theta: np.ndarray
) -> np.ndarray | float:
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
    f_values = np.zeros_like(x, dtype=float)
    for k in range(M + 1):
        f_values += beta[k] * bernstein_basis_polynomial(k, M, x)
    return f_values
