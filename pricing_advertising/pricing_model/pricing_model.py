import numpy as np
from pydantic import BaseModel, Field
from scipy.special import comb


class BernsteinPolynomialModel(BaseModel):
    """
    Pricing Model using Bernstein Basis Polynomials.
    """

    theta: np.ndarray = Field(description="Theta coefficients (non-negative).")

    @property
    def beta(self) -> np.ndarray:
        """
        Convert θ coefficients to β using cumulative sum.
        """
        m = len(self.theta) - 1
        _beta = np.zeros_like(self.theta)
        for k in range(m + 1):
            _beta[k] = np.sum(self.theta[k:])  # Reverse cumulative sum
        return _beta

    def bernstein_basis_polynomial(self, k: int, x: np.ndarray | float) -> np.ndarray | float:
        """
        Compute the Bernstein basis polynomial of degree `m` at position `k` for a given `x`.
        Parameters:
        -----------
        k : int
            The index of the Bernstein basis polynomial.
        x : np.ndarray or float
            The input value(s) at which to evaluate the polynomial. Can be a single float or an array of floats.
        Returns:
        --------
        np.ndarray or float
            The value(s) of the Bernstein basis polynomial at the given `x`.
        Notes:
        ------
        The Bernstein basis polynomial is defined as:
            B_k^m(x) = C(m, k) * x^k * (1 - x)^(m - k)
        where C(m, k) is the binomial coefficient "m choose k".
        """

        m = len(self.theta) - 1
        return comb(m, k) * (x**k) * ((1 - x) ** (m - k))

    def compute_pricing_function(self, x) -> np.ndarray | float:
        """
        Compute the pricing function using Bernstein basis polynomials.
        Parameters
        ----------
        x : array-like or float
            The input value(s) at which to evaluate the pricing function.
        Returns
        -------
        np.ndarray or float
            The computed pricing function values. The return type matches the type of `x`.
        Notes
        -----
        The pricing function is computed as a weighted sum of Bernstein basis polynomials,
        where the weights are given by the `beta` coefficients.
        """

        beta = self.beta
        m = len(beta) - 1
        f_values = np.zeros_like(x, dtype=float)
        for k in range(m + 1):
            f_values += beta[k] * self.bernstein_basis_polynomial(k, x)
        return f_values
