from abc import ABC, abstractmethod

from pricing_advertising.exceptions.exceptions import NotImplementedError


class ABCAcceptanceModel(ABC):
    """
    Abstract base class that defines the interface for the acceptance model
    """

    def accept(self, total_price: float) -> bool:
        result = self._accept(total_price)
        return result

    @abstractmethod
    def _accept(self, total_price: float) -> bool:
        raise NotImplementedError(message="This method is not implemented yet.")
