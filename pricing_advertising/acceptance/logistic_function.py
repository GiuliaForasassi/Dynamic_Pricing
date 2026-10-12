from datetime import date, timedelta

import numpy as np
from loguru import logger

from pricing_advertising.acceptance.acceptance import ABCAcceptanceModel
from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.requests.request import HotelOffer, Request
from pricing_advertising.utils.constants import BOOKING_WINDOW, HIGH_SEASON


# ------------- LOGISTIC ACCEPTANCE MODEL -------------
class LogisticAcceptanceModel(ABCAcceptanceModel):
    """Deterministic reference price + logistic acceptance curve.

    reference(night) = p0[category] * stag(night) * advance(night)
    P(accept) = 1 / (1 + exp(beta * (proposed_daily_price - mean_nights(reference)) / scale))

    `avg_prices` are the base prices p0 and `st_deviations` the price scale of the logistic curve.
    """

    def __init__(
        self,
        beta: float,
        avg_prices: dict,
        st_deviations: dict,
        *,
        high_season_multipliers: dict[int, float] | None = None,
        advance_max: float = 1.0,
        advance_min: float = 1.0,
        booking_window: int = BOOKING_WINDOW,
    ) -> None:
        super().__init__()
        # Beta parameter: how sensitive the probability is to the gap between proposed and reference price
        self.beta = beta
        # Dictionaries with base price and price scale per category
        self.avg_prices = avg_prices
        self.st_deviations = st_deviations
        self.high_season_multipliers = dict(high_season_multipliers or {})
        if any(
            not np.isfinite(multiplier) or multiplier <= 0
            for multiplier in self.high_season_multipliers.values()
        ):
            raise ValueError("High-season price multipliers must be finite and positive.")
        # Advance factor: linear decay from advance_max (same-day) to advance_min (booking_window days ahead)
        if not all(np.isfinite(factor) and factor > 0 for factor in (advance_max, advance_min)):
            raise ValueError("Advance factors must be finite and positive.")
        if advance_min > advance_max:
            raise ValueError("advance_min cannot be greater than advance_max.")
        if booking_window < 1:
            raise ValueError("The booking window must be at least one day.")
        self.advance_max = advance_max
        self.advance_min = advance_min
        self.booking_window = booking_window

    # ----------------- Acceptance Based on Total Price -----------------
    # Determines whether a given total price is accepted based on the logistic model.
    def _accept(self, proposed_price: float, request: HotelOffer) -> bool:
        # Extract relevant information from the request and calculate the proposed daily price.
        category = request.request.category
        los = max(1, int(request.days_stay))
        proposed_daily = proposed_price / los

        # Reference price of the offered stay: it depends on category, seasonality and advance
        reference_price = self.compute_reference_price(
            category,
            t_request=request.request.t_request,
            t_reservation=request.t_reservation,
            days_stay=los,
        )
        # Calculate the acceptance probability using the logistic function.
        acceptance_probability = self.logistic_function(
            reference_price=reference_price,
            proposed_price=proposed_daily,
            beta=self.beta,
            standard_deviation=self.st_deviations[category.value],
        )
        # Penalty if you propose an alternative: the acceptance is reduced by 20%
        if request.period_changed:
            acceptance_probability *= 0.8

        # Transform the probability into a boolean decision
        random_number = float(self.rng.random())
        # Compare the random number with the acceptance probability to make the decision
        accepted = random_number <= acceptance_probability
        # Return the final acceptance decision
        return accepted

    # ----------------- Acceptance Based on Daily Price -----------------
    # Method to determine acceptance based on the daily price rather than the total price.
    def _accept_daily_price(self, daily_price: float, request: HotelOffer) -> bool:
        return self._accept(daily_price * request.days_stay, request)

    # ----------------- Advance Factor -----------------
    # Linear decay: advance_max for same-day bookings, advance_min at booking_window days of advance or more.
    def advance_factor(self, advance: int) -> float:
        fraction = min(max(advance, 0), self.booking_window) / self.booking_window
        return self.advance_max - fraction * (self.advance_max - self.advance_min)

    # ----------------- Reference Price -----------------
    # Deterministic reference daily price: base price scaled, night by night, by seasonality and advance.
    def compute_reference_price(
        self,
        category: RoomCategory,
        *,
        t_request: date,
        t_reservation: date,
        days_stay: int = 1,
    ) -> float:
        if days_stay < 1:
            raise ValueError("The reference-price stay must contain at least one night.")
        season_multiplier = self.high_season_multipliers.get(category.value, 1.0)
        factor = 0.0
        for night in range(days_stay):
            night_date = t_reservation + timedelta(days=night)
            season = season_multiplier if night_date.month in HIGH_SEASON else 1.0
            factor += season * self.advance_factor((night_date - t_request).days)
        return self.avg_prices[category.value] * factor / days_stay

    # ----------------- Logistic Function -----------------
    # Acceptance probability: logistic of the gap between proposed and reference price (scalars or arrays).
    def logistic_function(
        self,
        proposed_price: float,
        reference_price: float,
        beta: float,
        standard_deviation: float,
    ) -> float:
        # x > 0 when the proposed price is above the reference price
        x = beta * (proposed_price - reference_price) / standard_deviation
        # clip in order to prevent overflow/underflow
        x = np.clip(x, -60, 60)
        return 1 / (1 + np.exp(x))


if __name__ == "__main__":
    avg_prices = {
        RoomCategory.CATEGORY1.value: 100,
        RoomCategory.CATEGORY2.value: 150,
        RoomCategory.CATEGORY3.value: 200,
    }
    st_deviations = {
        RoomCategory.CATEGORY1.value: 50,
        RoomCategory.CATEGORY2.value: 75,
        RoomCategory.CATEGORY3.value: 100,
    }
    beta = 0.1

    logistic_model = LogisticAcceptanceModel(
        beta=beta, avg_prices=avg_prices, st_deviations=st_deviations
    )
    request = Request(
        t_request=date.today(),
        t_reservation=date.today() + timedelta(days=30),
        days_stay=3,
        num_guests=2,
    )
    total_proposed_price = 110
    offer = HotelOffer(request, request.category, request.t_reservation, request.days_stay)
    acceptance = logistic_model.accept(total_proposed_price, offer)
    logger.info(f"Request: {request}")
    logger.info(f"Acceptance: {acceptance}")

    # Negative beta test
    try:
        LogisticAcceptanceModel(beta=-0.1, avg_prices=avg_prices, st_deviations=st_deviations)
        logger.warning("Negative beta test: Model initialized with negative beta!")
    except ValueError:
        logger.info("Correctly captured negative beta error.")

    # Zero beta test
    zero_beta_model = LogisticAcceptanceModel(
        beta=0, avg_prices=avg_prices, st_deviations=st_deviations
    )
    assert (
        zero_beta_model.logistic_function(
            proposed_price=110, reference_price=100, beta=0, standard_deviation=50
        )
        == 0.5
    ), "Zero beta should return 0.5 probability"

    # Acceptance probability test
    acceptance_prob = logistic_model.logistic_function(110, 100, 0.1, 50)
    assert 0 <= acceptance_prob <= 1, "Acceptance probability should be in [0,1]"

    # Reference price test
    reference_price = logistic_model.compute_reference_price(
        RoomCategory.CATEGORY1, t_request=date(2025, 1, 1), t_reservation=date(2025, 1, 5)
    )
    assert reference_price >= 0, "Reference price should be non-negative"

    # Invalid room category test
    try:
        logistic_model.compute_reference_price(
            None,  # type: ignore
            t_request=date(2025, 1, 1),
            t_reservation=date(2025, 1, 5),
        )
    except AttributeError:
        logger.info("Correctly caught invalid room category error.")

    logger.info("All tests passed successfully!")
