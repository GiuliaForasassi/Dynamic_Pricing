import numpy as np
from scipy.special import comb


class BernsteinProbModel:
    def __init__(self, theta: np.ndarray) -> None:
        """
        Initializes the object with the given theta values.

        Parameters
        ----------
        theta : np.ndarray
            A NumPy array of non-negative values representing the theta parameters.

        Raises
        ------
        TypeError
            If `theta` is not a NumPy array.
        ValueError
            If any value in `theta` is negative or if `theta` is empty.

        Notes
        -----
        The input array is converted to float type and stored as an instance attribute.
        """
        if not isinstance(theta, np.ndarray):
            raise TypeError("Theta values should be a NumPy array")
        if np.any(theta < 0):
            raise ValueError("Theta values should be non negative")
        if len(theta) == 0:
            raise ValueError("Theta values should not be empty")
        self.theta = theta.astype(float)  # Ensure theta is of type float

    @property
    def beta(self) -> np.ndarray:
        """
        Computes the cumulative sum of the reversed `theta` array.

        Returns:
            np.ndarray: The cumulative sum of the elements in `theta`, reversed, resulting in an array of the same shape as `theta`.
        """
        return np.cumsum(self.theta[::-1])

    def _bernstein_basis(self, k: int, x: np.ndarray | float) -> np.ndarray | float:
        """
        Compute the k-th Bernstein basis polynomial at point(s) x.

        The Bernstein basis polynomial of degree m is defined as:
            B_{k,m}(x) = C(m, k) * x^k * (1 - x)^(m - k)
        where C(m, k) is the binomial coefficient, and m is inferred from the length of self.theta.

        Args:
            k (int): The index of the Bernstein basis polynomial (0 <= k <= m).
            x (np.ndarray | float): The point(s) at which to evaluate the basis polynomial.

        Returns:
            np.ndarray | float: The value(s) of the k-th Bernstein basis polynomial at x.
        """
        m = len(self.theta) - 1
        return float(comb(m, k)) * (x**k) * ((1 - x) ** (m - k))

    def _bernstein_polynomial(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the Bernstein polynomial for a given input array.

        This method evaluates the Bernstein polynomial defined by the coefficients `self.beta`
        at the points specified by `x`. The polynomial is computed as a linear combination
        of Bernstein basis polynomials of degree `m`, where `m = len(self.beta) - 1`.

        Args:
            x (np.ndarray): Input array of points at which to evaluate the Bernstein polynomial.

        Returns:
            np.ndarray: The values of the Bernstein polynomial evaluated at each point in `x`.
        """
        beta = self.beta
        m = len(beta) - 1
        x_arr = np.asarray(x, dtype=float)
        y = np.zeros_like(x_arr, dtype=float)
        for k in range(m + 1):
            y += beta[k] * self._bernstein_basis(k, x_arr)
        return y

    def predict_increment_probs(self, occupancy: np.ndarray) -> np.ndarray:
        """
        Predicts the incremental probabilities based on the given occupancy values.

        This method applies a Bernstein polynomial transformation to the input occupancy array,
        then passes the result through a sigmoid function to obtain probabilities.

        Args:
            occupancy (np.ndarray): An array of occupancy values for which to predict incremental probabilities.

        Returns:
            np.ndarray: An array of predicted incremental probabilities corresponding to the input occupancy values.
        """
        z = self._bernstein_polynomial(occupancy)
        p = 1 / (1 + np.exp(-z))
        return p
