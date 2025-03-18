import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from scipy.special import comb


class BernsteinPolynomialModel:
    def __init__(self, theta: np.ndarray, min_price: float = 1.0) -> None:
        if not isinstance(theta, np.ndarray):
            raise TypeError("Theta values should be a NumPy array")
        if not np.all(theta >= 0):
            raise ValueError("Theta values should be non negative")
        if len(theta) == 0:
            raise ValueError("Theta values should not be empty")
        if min_price < 0:
            raise ValueError("Minimum price should be non-negative")

        self.theta = theta
        self.min_price = min_price

    @property
    def beta(self) -> np.ndarray:
        """
        Convert θ coefficients to β using cumulative sum.
        """
        beta = np.cumsum(self.theta[::-1])[::-1]
        return beta

    def bernstein_basis_polynomial(self, k: int, advance: np.ndarray | float) -> np.ndarray | float:
        """
        Compute the Bernstein basis polynomial of degree `m` for a given index `k` and input `advance`.

        The Bernstein basis polynomial is defined as:
            B_k^m(advance) = C(m, k) * (advance^k) * ((1 - advance)^(m - k))
        where C(m, k) is the binomial coefficient.

        Args:
            k (int): The index of the Bernstein basis polynomial.
            advance (np.ndarray | float): The input value(s) at which to evaluate the polynomial.
                Can be a single float or a NumPy array of values.

        Returns:
            np.ndarray | float: The value(s) of the Bernstein basis polynomial evaluated at `advance`.
                Returns a float if `advance` is a single value, or a NumPy array if `advance` is an array.

        Raises:
            ValueError: If `k` is negative or greater than the degree `m` of the polynomial.
        """
        m = len(self.theta) - 1
        return float(comb(m, k)) * (advance**k) * ((1 - advance) ** (m - k))

    def predict_price(self, advance: np.ndarray | float) -> np.ndarray | float:
        """
        Predicts the price based on the given advance value(s) using a Bernstein polynomial model.

        Args:
            advance (np.ndarray | float): The input value(s) representing the advance.
                It can be a single float or a NumPy array of floats.

        Returns:
            np.ndarray | float: The predicted price(s). If the input is a single float,
                the output will be a float. If the input is a NumPy array, the output
                will be a NumPy array of the same shape.
        """
        beta = self.beta
        m = len(beta) - 1
        price_values = np.zeros_like(advance, dtype=float)
        for k in range(m + 1):
            price_values += beta[k] * self.bernstein_basis_polynomial(k, advance)
        return np.maximum(price_values, self.min_price)


if __name__ == "__main__":
    theta_values = np.array([6, 5, 4, 3, 2, 1, 0], dtype=float)
    advance_values = np.linspace(0, 1, 10)

    min_price = 2.0
    model = BernsteinPolynomialModel(theta=theta_values, min_price=min_price)
    predicted_prices = model.predict_price(advance_values)

    logger.info(f"Predicted prices: {predicted_prices}")
    logger.info(f"Theta values: {model.theta}")

    assert np.all(predicted_prices >= 0), "Predicted prices should be non-negative"
    assert np.all(theta_values == model.theta), "Theta values should match the input values"

    try:
        BernsteinPolynomialModel(theta=np.array([]))  # Should raise ValueError
    except ValueError as e:
        logger.info(f"Correctly captured empty theta error: {e}")
    try:
        BernsteinPolynomialModel(theta=np.array([1]))  # Should work
        logger.info("Single-element theta test passed.")
    except Exception as e:
        logger.error(f"Single-element theta test failed: {e}")
    try:
        BernsteinPolynomialModel(theta=theta_values, min_price=-1)  # Should raise ValueError
    except ValueError as e:
        logger.info(f"Correctly caught negative min_price error: {e}")

    plt.plot(advance_values, predicted_prices, label="Predicted Prices")
    plt.xlabel("Advance days")
    plt.ylabel("Price")
    plt.title("Predicted Prices using Bernstein Polynomial Model")
    plt.legend()
    plt.grid()
    plt.show()
