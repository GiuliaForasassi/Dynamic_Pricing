from abc import ABC, abstractmethod
from datetime import date

import numpy as np

from pricing_advertising.clients.client import Client
from pricing_advertising.hotel.room import Room, RoomCategory
from pricing_advertising.utils.bernstein_polynomial import bernstein_monotonic_decreasing_polynomial


class PricingPolicy(ABC):
    # def __init__(self) -> None:
    # self.num_actions = ...
    # self.action_space = ....

    def __init__(self, start_simulation: date, end_simulation: date):
        super().__init__()
        self.start_simulation = start_simulation
        self.end_simulation = end_simulation

    def select_price(
        self, rooms: dict[RoomCategory, list[Room]], clients: list[Client]
    ) -> np.ndarray:
        num_categories = len(RoomCategory)
        coefficients = self.select_price_coefficients(rooms, clients)

        num_days = (self.end_simulation - self.start_simulation).days + 1
        prices = np.zeros((num_categories, num_days))
        for cat in range(num_categories):
            for day in range(num_days):
                prices[cat, day] = bernstein_monotonic_decreasing_polynomial(
                    x=day / num_days, theta=coefficients[cat]
                )

        return prices

    @abstractmethod
    def select_price_coefficients(
        self, rooms: dict[RoomCategory, list[Room]], clients: list[Client]
    ) -> np.ndarray:
        pass
