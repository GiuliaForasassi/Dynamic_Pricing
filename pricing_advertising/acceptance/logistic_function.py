import random

import numpy as np
from scipy.stats import norm

from pricing_advertising.acceptance.acceptance import ABCAcceptanceModel
from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.requests.request import Request


class LogisticAcceptanceModel(ABCAcceptanceModel):
    def __init__(self, beta: float, avg_prices: dict, variances: dict) -> None:
        self.beta = beta
        self.avg_prices = avg_prices
        self.variances = variances
        # can add season_factor to handle the seasonality of the prices

    def generate_reference_price(self, category: RoomCategory) -> float:
        avg_price = self.avg_prices[category]
        variance = self.variances[category]
        std = np.sqrt(variance)

        reference_price = norm.rvs(loc=avg_price, scale=std)
        return max(0, reference_price)

    def _accept(self, proposed_price: float, request: Request) -> bool:
        category = request.get_category()
        avg_price = self.avg_prices[category]
        variance = self.variances[category]

        reference_price = self.generate_reference_price(category=category)

        acceptance_probability = self._logistic_function(
            reference_price=reference_price,
            proposed_price=proposed_price,
            beta=self.beta,
            variance=variance,
            avg_price=avg_price,
        )
        random_number = random.random()
        return random_number <= acceptance_probability

    def _logistic_function(
        self,
        proposed_price: float,
        reference_price: float,
        beta: float,
        variance: float,
        avg_price: float,
    ) -> float:
        # Calculate the z-score respect to the average price: how many standard deviations away from the mean
        z_score = (proposed_price - avg_price) / np.sqrt(variance)

        # Calculate the correction factor based on the reference price
        correction_factor = (reference_price - avg_price) / np.sqrt(variance)

        # Calculate the acceptance probability
        acceptance_probability = 1 / (1 + np.exp(-beta * (z_score - correction_factor)))
        return acceptance_probability
