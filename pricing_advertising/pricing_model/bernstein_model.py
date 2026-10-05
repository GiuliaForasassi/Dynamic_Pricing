import numpy as np
from scipy.special import comb


# -------------- PROBABILISTIC BERNSTEIN MODEL ----------------
class BernsteinProbModel:
    # Initialize the Bernstein probabilistic model with the given theta parameters
    def __init__(self, theta: np.ndarray) -> None:
        # Validate the input theta array: it must be a non-empty NumPy array with non-negative values
        if not isinstance(theta, np.ndarray):
            raise TypeError("Theta values should be a NumPy array")
        if theta.ndim != 1 or theta.size == 0:
            raise ValueError("Theta must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(theta)) or np.any((theta < 0) | (theta > 1)):
            raise ValueError("Theta values must be finite and in [0, 1]")
        # Store the theta parameters as a float array
        self.theta = theta.astype(float)
        # Degree of the Bernstein polynomial (m = len(theta) - 1)
        self.m = len(self.theta) - 1
        # Dictionary to store precomputed binomial coefficients for efficiency
        self.comb_dict = {}
        # Precompute the binomial coefficients for the Bernstein basis polynomials
        for k in range(self.m + 1):
            # Precompute and store the binomial coefficient for the current k
            self.comb_dict[(self.m, k)] = float(comb(self.m, k))
        # Convert the dictionary of binomial coefficients to a NumPy array for vectorized operations
        self.comb_array = np.array(list(self.comb_dict.values()))
        # Array of indices k for the Bernstein basis polynomials
        self.k_array = np.arange(self.m + 1)
        # Array of m - k values for the Bernstein basis polynomials
        self.m_minus_k_array = np.array([self.m - k for k in range(self.m + 1)])

    # ---------------- Helper functions -------------------------
    @property
    def beta(self) -> np.ndarray:
        return np.cumsum(self.theta)

    # Compute the k-th Bernstein basis polynomial at the given point(s) x
    def _bernstein_basis(self, k: int, x: np.ndarray | float) -> np.ndarray | float:
        return self.comb_dict[(self.m, k)] * (x**k) * ((1 - x) ** (self.m - k))

    # ----------------- Optimization: Vectorized functions for Bernstein basis and polynomial -----------------
    def _bernstein_basis_vectorized(self, x: np.ndarray) -> np.ndarray:
        _x = x.reshape(-1, 1)
        return self.comb_array * (_x**self.k_array) * ((1 - _x) ** self.m_minus_k_array)

    def _bernstein_polynomial_vectorized(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        bb = self._bernstein_basis_vectorized(x_arr)
        y = bb.dot(self.beta)
        return y

    # ---------------- Bernstein polynomial evaluation -----------------
    # Compute the Bernstein polynomial at the given point(s) x using the precomputed beta coefficients
    def _bernstein_polynomial(self, x: np.ndarray) -> np.ndarray:
        # Take the precomputed beta coefficients
        beta = self.beta
        # Determine the degree of the polynomial based on the length of beta
        m = len(beta) - 1
        # Ensure the input is a NumPy array of type float
        x_arr = np.asarray(x, dtype=float)
        # Initialize the output array with zeros
        y = np.zeros_like(x_arr, dtype=float)
        # Loop over each term in the Bernstein polynomial and accumulate the result
        for k in range(m + 1):
            # Compute the k-th term of the Bernstein polynomial
            y += beta[k] * self._bernstein_basis(k, x_arr)  # bounded between 0 and 1
        return y  # bounded between 0 and m+1 (num coefficients)

    # ---------------- Incremental probability prediction -----------------
    # Predict the incremental probabilities for the given occupancy values using the Bernstein polynomial model
    def predict_increment_probs(self, occupancy: np.ndarray) -> np.ndarray:
        occupancy = np.asarray(occupancy, dtype=float)
        if not np.all(np.isfinite(occupancy)) or np.any((occupancy < 0) | (occupancy > 1)):
            raise ValueError("Occupancy values must be finite and in [0, 1]")
        # Evaluate the Bernstein polynomial at the given occupancy values
        z = self._bernstein_polynomial_vectorized(occupancy)
        # Scale the polynomial output to the [0, 1] range by dividing by the maximum possible value (m + 1)
        p = z / (self.m + 1)
        # Return the predicted incremental probabilities
        return p
