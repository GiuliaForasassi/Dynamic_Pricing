import random

import numpy as np

from pricing_advertising.acceptance.acceptance import ABCAcceptanceModel


class LogisticAcceptanceModel(ABCAcceptanceModel):
    def _accept(self, proposed_price: float, reference_price: float, beta: float) -> bool:
        """
        Determines whether a given total price is accepted based on a threshold.

        Args:
            proposed_price (float): The price being proposed.
            reference_price (float): The reference price to compare against.
            beta (float): The sensitivity parameter that controls the steepness of the logistic function.

        Returns:
            bool: True if the total price is accepted, False otherwise.
        """
        acceptance_probability = self.logistic_function(proposed_price, reference_price, beta)
        random_number = random.random()
        return random_number <= acceptance_probability

    def logistic_function(
        self, proposed_price: float, reference_price: float, beta: float
    ) -> float:
        """
        Calculate the acceptance probability using a logistic function.

        Parameters:
        proposed_price (float): The price being proposed.
        reference_price (float): The reference price to compare against.
        beta (float): The sensitivity parameter that controls the steepness of the logistic function.

        Returns:
        float: The probability of acceptance, ranging from 0 to 1.
        """
        acceptance_probability = 1 / (1 + np.exp(-beta * (proposed_price - reference_price)))
        return acceptance_probability
