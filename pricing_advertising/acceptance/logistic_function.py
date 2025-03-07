import random

import numpy as np

from pricing_advertising.acceptance.acceptance import ABCAcceptanceModel


class LogisticAcceptanceModel(ABCAcceptanceModel):
    def __init__(self, reference_price: float, beta: float) -> None:
        self.reference_price = reference_price
        self.beta = beta

    def _accept(self, proposed_price: float) -> bool:
        """
        Determines whether a given total price is accepted based on a threshold.

        Args:
            proposed_price (float): The price being proposed.
            reference_price (float): The reference price to compare against.
            beta (float): The sensitivity parameter that controls the steepness of the logistic function.

        Returns:
            bool: True if the proposed price is accepted, False otherwise.
        """
        acceptance_probability = self._logistic_function(
            reference_price=self.reference_price, proposed_price=proposed_price, beta=self.beta
        )
        random_number = random.random()
        return random_number <= acceptance_probability

    def _logistic_function(
        self, proposed_price: float, reference_price: float, beta: float
    ) -> float:
        """
        Calculate the acceptance probability using a logistic function.

        Parameters:
        proposed_price (float): The price being proposed.
        reference_price (float): The price that a customer expects to pay based on their own perception of value, past experiences, or competitor prices.
        beta (float): The sensitivity parameter that controls the steepness of the logistic function.

        Returns:
        float: The probability of acceptance, ranging from 0 to 1.
        """
        acceptance_probability = 1 / (1 + np.exp(-beta * (proposed_price - reference_price)))
        return acceptance_probability
