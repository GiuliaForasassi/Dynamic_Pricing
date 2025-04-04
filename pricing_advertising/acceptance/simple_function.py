from random import random

from pricing_advertising.acceptance.acceptance import ABCAcceptanceModel
from pricing_advertising.requests.request import Request


class SimpleAcceptanceModel(ABCAcceptanceModel):
    def __init__(self, threshold: dict) -> None:
        self.threshold = threshold

    def _accept(self, total_price: float, request: Request) -> bool:
        """
        Determines whether a given total price is accepted based on a threshold.

        Args:
            total_price (float): The total price to evaluate.

        Returns:
            bool: True if the total price is accepted, False otherwise.
        """

        category = request.category
        total_price_threshold = self.threshold[category]

        if total_price < total_price_threshold:
            return random() < 0.8
        else:
            return random() < 0.2

    def _accept_daily_price(self, daily_price: float, request: Request) -> None:
        pass
