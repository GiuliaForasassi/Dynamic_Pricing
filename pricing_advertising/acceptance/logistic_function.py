from datetime import date, timedelta

import numpy as np
from loguru import logger

from pricing_advertising.acceptance.acceptance import ABCAcceptanceModel
from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.requests.request import HotelOffer, Request
from pricing_advertising.utils.constants import HIGH_SEASON


# ------------- LOGISTIC ACCEPTANCE MODEL -------------
class LogisticAcceptanceModel(ABCAcceptanceModel):
    def __init__(
        self,
        beta: float,
        avg_prices: dict,
        st_deviations: dict,
        *,
        high_season_multipliers: dict[int, float] | None = None,
    ) -> None:
        super().__init__()
        # Beta parameter: check how sensitive the probability is to the difference between the proposed price and the reference price
        self.beta = beta
        # Dictionaries with avg price and variance per category
        self.avg_prices = avg_prices
        self.st_deviations = st_deviations
        self.high_season_multipliers = dict(high_season_multipliers or {})
        if any(
            not np.isfinite(multiplier) or multiplier <= 0
            for multiplier in self.high_season_multipliers.values()
        ):
            raise ValueError("High-season price multipliers must be finite and positive.")

    # ----------------- Acceptance Based on Total Price -----------------
    # Determines whether a given total price is accepted based on the logistic model.
    def _accept(self, proposed_price: float, request: HotelOffer) -> bool:
        # Extract relevant information from the request and calculate the proposed daily price.
        category = request.request.category
        los = max(1, int(request.days_stay))
        proposed_daily = proposed_price / los
        # Retrieve the standard deviation and average price for the requested category.
        st_deviations = self.st_deviations[category.value]
        avg_price = self.avg_prices[category.value]

        # Generate the reference price for the client for the requested category
        reference_price = self.generate_reference_price(
            category=category,
            t_reservation=request.t_reservation,
            days_stay=los,
        )
        # Calculate the acceptance probability using the logistic function.
        acceptance_probability = self.logistic_function(
            reference_price=reference_price,
            proposed_price=proposed_daily,
            beta=self.beta,
            standard_deviation=st_deviations,
            avg_price=avg_price,
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

    # ----------------- Reference Price Generation -----------------
    # Method to generate a reference price for a given room category based on the average price and standard deviation.
    def generate_reference_price(
        self,
        category: RoomCategory,
        *,
        t_reservation: date | None = None,
        days_stay: int = 1,
    ) -> float:
        if days_stay < 1:
            raise ValueError("The reference-price stay must contain at least one night.")
        # Retrieve the average price and standard deviation for the given room category.
        avg_price = self.avg_prices[category.value]
        std = self.st_deviations[category.value]
        if t_reservation is not None:
            high_season_nights = sum(
                (t_reservation + timedelta(days=night)).month in HIGH_SEASON
                for night in range(days_stay)
            )
            multiplier = self.high_season_multipliers.get(category.value, 1.0)
            avg_price *= 1.0 + (multiplier - 1.0) * high_season_nights / days_stay
        # Generate the reference price using a normal distribution centered around the average price with the given standard deviation.
        reference_price = float(self.rng.normal(loc=avg_price, scale=std))
        # Ensure the reference price is non-negative before returning.
        return max(0, reference_price)

    # ----------------- Logistic Function -----------------
    # Method to calculate the acceptance probability using the logistic function.
    def logistic_function(
        self,
        proposed_price: float,
        reference_price: float,
        beta: float,
        standard_deviation: float,
        avg_price: float,
    ) -> float:
        # Calculate the z-score respect to the average price: how many standard deviations away from the mean
        z_score = (proposed_price - avg_price) / standard_deviation

        # Calculate the correction factor based on the reference price
        correction_factor = (reference_price - avg_price) / standard_deviation

        # x measures how the proposed price is below or above the reference price
        x = beta * (z_score - correction_factor)
        # clippo in order to prevent overflow/underflow
        x = np.clip(x, -60, 60)
        # Calculate the acceptance probability
        acceptance_probability = 1 / (1 + np.exp(x))
        return acceptance_probability


if __name__ == "__main__":
    avg_prices = {
        RoomCategory.CATEGORY1: 100,
        RoomCategory.CATEGORY2: 150,
        RoomCategory.CATEGORY3: 200,
    }
    st_deviations = {
        RoomCategory.CATEGORY1: 50,
        RoomCategory.CATEGORY2: 75,
        RoomCategory.CATEGORY3: 100,
    }
    beta = 0.1

    logistic_model = LogisticAcceptanceModel(
        beta=beta, avg_prices=avg_prices, st_deviations=st_deviations
    )
    request = Request(
        t_request=date.today(), t_reservation=date(2025, 3, 30), days_stay=3, num_guests=2
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
            proposed_price=110, reference_price=100, beta=0, standard_deviation=50, avg_price=100
        )
        == 0.5
    ), "Zero beta should return 0.5 probability"

    # Acceptance probability test
    acceptance_prob = logistic_model.logistic_function(110, 100, 0.1, 50, 100)
    assert 0 <= acceptance_prob <= 1, "Acceptance probability should be in [0,1]"

    # Reference price test
    reference_price = logistic_model.generate_reference_price(RoomCategory.CATEGORY1)
    assert reference_price >= 0, "Reference price should be non-negative"

    # Invalid room category test
    try:
        logistic_model.generate_reference_price(None)  # type: ignore
    except KeyError:
        logger.info("Correctly caught invalid room category error.")

    logger.info("All tests passed successfully!")
