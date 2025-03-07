from abc import ABC, abstractmethod

from pricing_advertising.requests.request import Request


class ABCAcceptanceModel(ABC):
    """
    Abstract base class that defines the interface for the acceptance model
    """

    def accept(self, total_price: float, request: Request) -> bool:
        """
        Determines whether the given total price is acceptable.

        Args:
            total_price (float): The total price to evaluate.

        Returns:
            bool: True if the total price is acceptable, False otherwise.
        """
        result = self._accept(total_price, request)
        return result

    @abstractmethod
    def _accept(self, total_price: float, request: Request) -> bool:
        raise NotImplementedError("The _accept method must be implemented by the subclass.")

    def accept_daily_price(self, daily_price: float, request: Request) -> bool:
        """
        Determines whether the given daily price is acceptable.

        Args:
            daily_price (float): The daily price to evaluate.

        Returns:
            bool: True if the daily price is acceptable, False otherwise.
        """
        result = self._accept_daily_price(daily_price, request)
        return result

    @abstractmethod
    def _accept_daily_price(self, daily_price: float, request: Request) -> bool:
        raise NotImplementedError(
            "The _accept_daily_price method must be implemented by the subclass."
        )
