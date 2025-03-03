from abc import ABC, abstractmethod


class ABCAcceptanceModel(ABC):
    """
    Abstract base class that defines the interface for the acceptance model
    """

    def accept(self, total_price: float) -> bool:
        """
        Determines whether the given total price is acceptable.

        Args:
            total_price (float): The total price to evaluate.

        Returns:
            bool: True if the total price is acceptable, False otherwise.
        """
        result = self._accept(total_price)
        return result

    @abstractmethod
    def _accept(self, total_price: float) -> bool:
        raise NotImplementedError("The _accept method must be implemented by the subclass.")
