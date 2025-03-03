"""
import random
from datetime import date

import numpy as np

from pricing_advertising.hotel.room import RoomCategory
from pricing_advertising.policies.pricing.policy import PricingPolicy
from pricing_advertising.utils.constants import NUM_COEFFICIENTS, VARIABILITY_MIN_MAX_PRICE

 # TODO: DA SISTEMARE
class RandomPricingPolicy(PricingPolicy):
    def __init__(self, start_simulation: date, end_simulation: date) -> None:
        super().__init__(start_simulation, end_simulation)

    def select_price_coefficients(self, observation) -> np.ndarray:
        num_categories = len(RoomCategory)
        prices_coefficients = np.zeros((num_categories, NUM_COEFFICIENTS))
        for cat in range(num_categories):
            for k in range(NUM_COEFFICIENTS):
                # kth coefficient for the category
                prices_coefficients[cat, k] = max(0, generate_random_price(100))

        return prices_coefficients  # prices.shape == (num_categories, num_coefficents)


def generate_random_price(base_price: float) -> float:
    min_price = base_price * (1 - VARIABILITY_MIN_MAX_PRICE)
    max_price = base_price * (1 + VARIABILITY_MIN_MAX_PRICE)
    random_price = random.uniform(min_price, max_price)
    return round(random_price, 2)"""
