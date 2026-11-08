from dataclasses import dataclass, field
from datetime import date, timedelta
from math import exp, expm1, isfinite, log, log1p

import numpy as np
from numpy.typing import NDArray

from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.utils.constants import HIGH_SEASON, LOGISTIC_ACCEPTANCE_AVG_PRICES


@dataclass(frozen=True)
class CancellationModel:
    daily_probability: float = 0.005
    price_sensitivity: float = 1.0
    high_season_multiplier: float = 0.8
    reference_prices: tuple[float, ...] = field(
        default_factory=lambda: tuple(
            float(LOGISTIC_ACCEPTANCE_AVG_PRICES[category.value]) for category in RoomCategory
        )
    )
    timing_strength: float = 0.3
    timing_decay_days: float = 3.0

    def __post_init__(self) -> None:
        if not isfinite(self.daily_probability) or not 0 <= self.daily_probability <= 1:
            raise ValueError("Daily cancellation probability must be finite and in [0, 1].")
        if not isfinite(self.price_sensitivity) or self.price_sensitivity < 0:
            raise ValueError("Price sensitivity must be finite and non-negative.")
        if not isfinite(self.high_season_multiplier) or not 0 < self.high_season_multiplier <= 1:
            raise ValueError("High-season multiplier must be finite and in (0, 1].")
        if not isfinite(self.timing_strength) or self.timing_strength < 0:
            raise ValueError("Timing strength must be finite and non-negative.")
        if not isfinite(self.timing_decay_days) or self.timing_decay_days <= 0:
            raise ValueError("Timing decay days must be finite and positive.")
        references = tuple(self.reference_prices)
        if len(references) != len(RoomCategory) or any(
            not isfinite(price) or price <= 0 for price in references
        ):
            raise ValueError("Reference prices must be finite and positive for every category.")
        object.__setattr__(self, "reference_prices", references)

    def probability(
        self, category: RoomCategory, check_in: date, nightly_prices: tuple[float, ...]
    ) -> float:
        if not nightly_prices or any(not isfinite(price) or price < 0 for price in nightly_prices):
            raise ValueError("Nightly prices must be non-empty, finite and non-negative.")
        if self.daily_probability in (0, 1):
            return self.daily_probability
        mean_price = sum(price / len(nightly_prices) for price in nightly_prices)
        price_ratio = mean_price / self.reference_prices[category.value - 1]
        high_season_fraction = sum(
            (check_in + timedelta(days=night)).month in HIGH_SEASON
            for night in range(len(nightly_prices))
        ) / len(nightly_prices)
        price_effect = self.price_sensitivity * (price_ratio - 1) if self.price_sensitivity else 0.0
        log_odds = (
            log(self.daily_probability)
            - log1p(-self.daily_probability)
            - price_effect
            + high_season_fraction * log(self.high_season_multiplier)
        )
        if log_odds >= 0:
            return 1 / (1 + exp(-log_odds))
        odds = exp(log_odds)
        return odds / (1 + odds)

    def _timing_weights(self, eligible_days: int) -> NDArray[np.float64]:
        offsets = np.arange(eligible_days, dtype=np.float64)
        with np.errstate(over="ignore"):
            early = np.exp(-offsets / self.timing_decay_days)
        baseline = 1 / (1 + self.timing_strength)
        weights = baseline + (self.timing_strength * baseline) * (early + early[::-1])
        return weights * (eligible_days / weights.sum())

    def sample_waiting_days(
        self, probability: float, advance: int, rng: np.random.Generator
    ) -> int | None:
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Cancellation probability must be finite and in [0, 1].")
        if probability == 0 or advance <= 1:
            return None
        if self.timing_strength == 0:
            waiting_days = int(rng.geometric(probability))
            return waiting_days if waiting_days < advance else None
        if probability == 1:
            return 1
        eligible_days = advance - 1
        log_survival = log1p(-probability)
        total_probability = -expm1(eligible_days * log_survival)
        draw = rng.random()
        if draw >= total_probability:
            return None
        cumulative_weights = np.cumsum(self._timing_weights(eligible_days))
        cumulative_probabilities = -np.expm1(log_survival * cumulative_weights)
        cumulative_probabilities[-1] = total_probability
        return int(np.searchsorted(cumulative_probabilities, draw, side="right")) + 1


@dataclass(frozen=True)
class Booking:
    booking_id: int
    booked_on: date
    category: RoomCategory
    check_in: date
    nightly_prices: tuple[float, ...]
    total_price: float
    cancellation_probability: float
    cancellation_day: date | None = None
    reward_accounted: bool = True

    @property
    def days_stay(self) -> int:
        return len(self.nightly_prices)
