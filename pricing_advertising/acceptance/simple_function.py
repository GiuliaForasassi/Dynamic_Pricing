from pricing_advertising.acceptance.acceptance import ABCAcceptanceModel
from pricing_advertising.requests.request import HotelOffer


class SimpleAcceptanceModel(ABCAcceptanceModel):
    def __init__(self, threshold: dict) -> None:
        super().__init__()
        self.threshold = threshold

    def _accept(self, total_price: float, request: HotelOffer) -> bool:
        category = request.request.category
        total_price_threshold = self.threshold[category]

        probability = 0.8 if total_price < total_price_threshold else 0.2
        if request.period_changed:
            probability *= 0.8
        return bool(self.rng.random() < probability)

    def _accept_daily_price(self, daily_price: float, request: HotelOffer) -> bool:
        return self._accept(daily_price * request.days_stay, request)
