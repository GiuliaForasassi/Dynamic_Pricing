from random import random

from pricing_advertising.acceptance.acceptance import ABCAcceptanceModel
from pricing_advertising.utils.constants import TOTAL_PRICE_THRESHOLD


class SimpleAcceptanceModel(ABCAcceptanceModel):
    def _accept(self, total_price: float) -> bool:
        """
        Determines whether a given total price is accepted based on a threshold.

        Args:
            total_price (float): The total price to evaluate.

        Returns:
            bool: True if the total price is accepted, False otherwise.
        """
        if total_price < TOTAL_PRICE_THRESHOLD:
            return random() < 0.8
        else:
            return random() < 0.2

    def _accept_daily_price(self, daily_price: float) -> None:
        pass
