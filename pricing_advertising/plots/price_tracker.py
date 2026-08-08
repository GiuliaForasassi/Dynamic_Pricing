from __future__ import annotations

import csv
from io import StringIO
from typing import TYPE_CHECKING

import mlflow
import numpy as np

if TYPE_CHECKING:
    from datetime import date

    from pricing_advertising.requests.request import Request


class PriceTracker:
    """Accumulates per-day average room price and logs it to mlflow, per training split.

    Two metrics are produced (keys suffixed with `split`, e.g. "train"/"val"):
      - "price/by_training_timestep/{split}": mean price over an episode (or, for eval,
        over a whole eval round), logged against a caller-supplied `step` (typically
        `num_timesteps`, so train/val curves share the same x-axis and can be
        overlaid/compared in the MLflow UI). See `end_episode`/`log_step_metric`.
      - "price/by_reservation_day/{split}": one point per simulation day (0-365), averaged
        across all episodes seen so far, logged once `finalize_training` is
        called (i.e. at the end of training).

    Prices are attributed to actual occupancy days (t_reservation), not request arrival days.
    A CSV artifact (`day,price`) is also logged per episode.
    """

    def __init__(self, split: str, log_to_mlflow: bool = True) -> None:
        self.split = split  # train or val
        self.log_to_mlflow = log_to_mlflow

        self._episode_id: int = 0
        self._start_simulation: date | None = None
        self._simulation_days: int | None = None  # actual simulation days (0-365)

        # Per-episode price tracking
        self._episode_price_sum_by_day: np.ndarray | None = None
        self._episode_price_count_by_day: np.ndarray | None = None

        # Across-episode aggregation
        self._day_price_sum: np.ndarray | None = None
        self._day_price_count: np.ndarray | None = None

    def set_simulation_params(self, start_simulation: date, simulation_days: int) -> None:
        """Set simulation parameters needed for day indexing.

        Args:
            start_simulation: Start date of simulation.
            simulation_days: Actual simulation days (0-365).
        """
        self._start_simulation = start_simulation
        self._simulation_days = simulation_days
        if self._episode_price_sum_by_day is None:
            self._episode_price_sum_by_day = np.zeros(simulation_days, dtype=float)
            self._episode_price_count_by_day = np.zeros(simulation_days, dtype=int)

    def record_request_prices(
        self,
        final_requests: np.ndarray,
        final_prices: np.ndarray,
    ) -> None:
        """Record prices for accepted requests, attributed to their occupancy days.

        Args:
            final_requests: Array of accepted Request objects.
            final_prices: Array of prices (one per day per request).
        """
        if self._episode_price_sum_by_day is None or self._start_simulation is None:
            return

        # Reconstruct mapping: final_prices are ordered by request, with days_stay consecutive prices
        price_idx = 0
        for req in final_requests:
            from typing import cast

            from pricing_advertising.requests.request import Request

            req = cast("Request", req)
            num_days = req.days_stay

            # Extract this request's prices
            if price_idx + num_days <= len(final_prices):
                req_prices = final_prices[price_idx : price_idx + num_days]

                # Attribute each price to its actual occupancy day (only within simulation period)
                start_idx = (req.t_reservation - self._start_simulation).days
                for day_offset, price in enumerate(req_prices):
                    day_idx = start_idx + day_offset
                    if 0 <= day_idx < self._simulation_days:
                        self._episode_price_sum_by_day[day_idx] += price
                        self._episode_price_count_by_day[day_idx] += 1

                price_idx += num_days

    def get_current_day_avg_price(self, today: date) -> float:
        """Get average price for the given day (occupancy day, not arrival day)."""
        if self._start_simulation is None or self._episode_price_sum_by_day is None:
            return 0.0

        current_day_index = (today - self._start_simulation).days
        if (
            0 <= current_day_index < len(self._episode_price_sum_by_day)
            and self._episode_price_count_by_day[current_day_index] > 0
        ):
            return float(
                self._episode_price_sum_by_day[current_day_index]
                / self._episode_price_count_by_day[current_day_index]
            )
        return 0.0

    def end_episode(self) -> float | None:
        """Flush the current episode: write its CSV artifact, accumulate per-day
        values into running averages, and return the episode's mean price
        (or None if no data was recorded).
        """
        if (
            self._episode_price_sum_by_day is None
            or self._episode_price_count_by_day is None
            or np.sum(self._episode_price_count_by_day) == 0
        ):
            return None

        # Calculate episode mean price (only for days with prices)
        valid_days = self._episode_price_count_by_day > 0
        if not np.any(valid_days):
            return None

        mean_price = float(
            np.sum(self._episode_price_sum_by_day[valid_days])
            / np.sum(self._episode_price_count_by_day[valid_days])
        )

        # Save a CSV artifact on MLflow
        if self.log_to_mlflow:
            buf = StringIO()
            writer = csv.writer(buf)
            writer.writerow(["day", "price"])
            for day_idx in range(len(self._episode_price_sum_by_day)):
                if self._episode_price_count_by_day[day_idx] > 0:
                    daily_avg = (
                        self._episode_price_sum_by_day[day_idx]
                        / self._episode_price_count_by_day[day_idx]
                    )
                    writer.writerow([day_idx, daily_avg])
            mlflow.log_text(
                buf.getvalue(),
                f"episodes/{self.split}/episode_{self._episode_id:05d}_price.csv",
            )

        # Accumulate into across-episode statistics
        if self._day_price_sum is None or self._day_price_count is None:
            self._day_price_sum = np.zeros_like(self._episode_price_sum_by_day, dtype=float)
            self._day_price_count = np.zeros_like(self._episode_price_count_by_day, dtype=int)

        self._day_price_sum += self._episode_price_sum_by_day
        self._day_price_count += self._episode_price_count_by_day

        # Reset for next episode
        self._episode_price_sum_by_day.fill(0)
        self._episode_price_count_by_day.fill(0)
        self._episode_id += 1

        return mean_price

    def log_step_metric(self, value: float, step: int) -> None:
        """Log a single scalar (e.g. mean episode price) against `step`
        (typically `num_timesteps`), so train/val curves are directly
        comparable in the MLflow UI.
        """
        if self.log_to_mlflow:
            mlflow.log_metric(f"price/by_training_timestep/{self.split}", value, step=step)

    def finalize_training(self) -> None:
        # Closes the latest episode if there is one still in progress
        if (
            self._episode_price_count_by_day is not None
            and np.sum(self._episode_price_count_by_day) > 0
        ):
            self.end_episode()

        if not self.log_to_mlflow or self._day_price_sum is None or self._day_price_count is None:
            return

        # Calculates the avg price taking into account all episodes and send this data to MLflow
        for day in range(len(self._day_price_sum)):
            if self._day_price_count[day] > 0:
                mlflow.log_metric(
                    f"price/by_reservation_day/{self.split}",
                    float(self._day_price_sum[day] / self._day_price_count[day]),
                    step=day,
                )
