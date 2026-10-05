from abc import ABC, abstractmethod

import numpy as np

from pricing_advertising.requests.request import HotelOffer


# ----------------- ABC ACCEPTANCE MODEL -----------------
class ABCAcceptanceModel(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.rng = np.random.default_rng()

    def accept(self, total_price: float, request: HotelOffer) -> bool:
        result = self._accept(total_price, request)
        return result

    @abstractmethod
    def _accept(self, total_price: float, request: HotelOffer) -> bool:
        raise NotImplementedError("The _accept method must be implemented by the subclass.")

    def accept_daily_price(self, daily_price: float, request: HotelOffer) -> bool:
        result = self._accept_daily_price(daily_price, request)
        return result

    @abstractmethod
    def _accept_daily_price(self, daily_price: float, request: HotelOffer) -> bool:
        raise NotImplementedError(
            "The _accept_daily_price method must be implemented by the subclass."
        )
