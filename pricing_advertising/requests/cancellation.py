from dataclasses import dataclass, field
from datetime import date, timedelta
from math import exp, expm1, isfinite, log, log1p, log2, log10

import numpy as np
from numpy.typing import NDArray

from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.utils.constants import HIGH_SEASON, LOGISTIC_ACCEPTANCE_AVG_PRICES


@dataclass(frozen=True)
class CancellationModel:
    base_daily_cancellation_prob: float = 0.005  # Base daily cancellation probability in low season, when advance is 10 days (because of log10) and price is at reference level
    high_season_multiplier: float = 0.8
    timing_strength: float = 0.3
    timing_decay_days: float = 3.0

    def __post_init__(self) -> None:
        if (
            not isfinite(self.base_daily_cancellation_prob)
            or not 0 <= self.base_daily_cancellation_prob <= 1
        ):
            raise ValueError("Daily cancellation probability must be finite and in [0, 1].")
        if not isfinite(self.high_season_multiplier) or not 0 < self.high_season_multiplier <= 1:
            raise ValueError("High-season multiplier must be finite and in (0, 1].")
        if not isfinite(self.timing_strength) or self.timing_strength < 0:
            raise ValueError("Timing strength must be finite and non-negative.")
        if not isfinite(self.timing_decay_days) or self.timing_decay_days <= 0:
            raise ValueError("Timing decay days must be finite and positive.")

    def cancellation_probability(
        self,
        category: RoomCategory,
        check_in: date,
        nightly_prices: list[float],
        advance: int,
    ) -> float:
        if not nightly_prices or any(not isfinite(price) or price < 0 for price in nightly_prices):
            raise ValueError("Nightly prices must be non-empty, finite and non-negative.")
        if self.base_daily_cancellation_prob in (0, 1):
            return self.base_daily_cancellation_prob
        mean_reservation_price = sum(nightly_prices) / len(nightly_prices)
        reference_price = LOGISTIC_ACCEPTANCE_AVG_PRICES[category.value]
        high_season_fraction = sum(
            (check_in + timedelta(days=night)).month in HIGH_SEASON
            for night in range(len(nightly_prices))
        ) / len(nightly_prices)

        # Compute the multiplicative factors for price, seasonality, and advance
        k_p = exp(-(mean_reservation_price - reference_price) / reference_price)
        k_s = 1 - (1 - self.high_season_multiplier) * high_season_fraction
        k_a = advance

        k = k_p * k_s * k_a

        # Compute q (positive value representing the odds of cancellation) to avoid messing directly with probabilities
        q = k * self.base_daily_cancellation_prob / (1 - self.base_daily_cancellation_prob)
        cancellation_prob = q / (1 + q)
        return cancellation_prob

    def sample_cancellation_waiting_days(self, advance: int, rng: np.random.Generator) -> int:
        n_eligible_cancellation_days = advance - 1
        timing_weights = self._timing_weights(n_eligible_cancellation_days)
        cancellation_waiting_days = rng.choice(len(timing_weights), p=timing_weights) + 1
        return cancellation_waiting_days

    def _timing_weights(self, n_eligible_cancellation_days: int) -> NDArray[np.float64]:
        offsets = np.arange(n_eligible_cancellation_days, dtype=np.float64)
        early = np.exp(-offsets / self.timing_decay_days)  # exponential decay
        # idea: each days has a base weight (min_weight), and we add extra weight for early and late days based on the exponential decay
        min_weight = 1 / (1 + self.timing_strength)
        more_weight_for_nearby_days = (self.timing_strength * min_weight) * early
        more_weight_for_late_days = (self.timing_strength * min_weight) * early[::-1]
        weights = min_weight + more_weight_for_nearby_days + more_weight_for_late_days
        normalized_weights = weights / weights.sum()
        return normalized_weights


@dataclass(frozen=True)
class Booking:
    booking_id: int
    booked_on: date
    category: RoomCategory
    check_in: date
    nightly_prices: list[float]
    total_price: float
    cancellation_probability: float
    cancellation_day: date | None = None
    reward_accounted: bool = True

    @property
    def days_stay(self) -> int:
        return len(self.nightly_prices)
