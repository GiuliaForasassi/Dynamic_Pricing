from __future__ import annotations

import csv
from io import StringIO

import mlflow
import numpy as np


# ---------------- REQUEST TRACKER ------------------
# Tracks requests by 2 different points of view:
# 1. by_request_day: counts requests arriving on each day.
# 2. by_stay_day: counts requests covering each night of the stay.
# Example: taking a request the day 5 for a 3-night stay with check-in the day 10
# by_request_day will count this request on day 5 --> +1 at the day 5
# by_stay_day will count this request on days 10, 11, and 12 --> +1 at each of these days
class RequestTracker:
    def __init__(self, split: str, log_to_mlflow: bool = True) -> None:
        self.split = split
        self.log_to_mlflow = log_to_mlflow

        self._simulation_days: int | None = None
        self._next_day = 0
        self._episode_id = 0
        self._completed_episodes = 0
        self._pending_requests = 0
        self._pending_episodes = 0

        self._arrivals: np.ndarray | None = None
        self._stays: np.ndarray | None = None
        self._arrivals_sum: np.ndarray | None = None
        self._stays_sum: np.ndarray | None = None

    # -------------------- RECORDING EACH DAY--------------------
    def record_day(
        self,
        *,
        day_index: int,
        simulation_days: int,
        num_requests: int,
        request_stays: list[tuple[int, int]],
    ) -> None:
        # - Validate input parameters
        if simulation_days <= 0:
            raise ValueError("Simulation days must be positive.")

        if self._simulation_days is not None and simulation_days != self._simulation_days:
            raise ValueError("Simulation duration changed: use a separate tracker.")

        if day_index != self._next_day or day_index >= simulation_days:
            raise ValueError("Missing, duplicate, or out-of-sequence day.")

        if num_requests != len(request_stays):
            raise ValueError("Number of requests and stays inconsistent.")

        for check_in, nights in request_stays:
            if check_in < day_index or nights <= 0:
                raise ValueError("Invalid stay.")

        # - Initialize arrays if this is the first day of the simulation
        if self._simulation_days is None:
            self._simulation_days = simulation_days
            self._arrivals = np.zeros(simulation_days, dtype=np.int64)
            self._stays = np.zeros(simulation_days, dtype=np.int64)
            self._arrivals_sum = np.zeros(simulation_days, dtype=np.int64)
            self._stays_sum = np.zeros(simulation_days, dtype=np.int64)

        assert self._arrivals is not None
        assert self._stays is not None

        # - Record arrivals and stays for the current day
        self._arrivals[day_index] = num_requests

        for check_in, nights in request_stays:
            stop = min(check_in + nights, simulation_days)
            if check_in < stop:
                self._stays[check_in:stop] += 1

        self._next_day += 1

    # -------------------- END OF EPISODE --------------------
    # End of episode and update statistics
    def end_episode(self) -> None:
        # - Validations
        if self._next_day == 0:
            return

        if self._next_day != self._simulation_days:
            raise ValueError("Cannot close an incomplete episode.")

        assert self._arrivals is not None
        assert self._stays is not None
        assert self._arrivals_sum is not None
        assert self._stays_sum is not None

        # - Log episode data to MLflow if enabled
        if self.log_to_mlflow:
            buf = StringIO()
            writer = csv.writer(buf)
            writer.writerow(["day", "requests_arriving", "requests_covering_night"])
            writer.writerows(
                (day + 1, int(arrivals), int(stays))
                for day, (arrivals, stays) in enumerate(
                    zip(self._arrivals, self._stays, strict=True)
                )
            )
            mlflow.log_text(
                buf.getvalue(),
                (f"episodes/{self.split}/episode_{self._episode_id:05d}_request_counts.csv"),
            )

        # - Update cumulative statistics
        self._arrivals_sum += self._arrivals
        self._stays_sum += self._stays
        self._completed_episodes += 1

        # - Update pending requests and episodes for the next logging checkpoint
        self._pending_requests += int(self._arrivals.sum())
        self._pending_episodes += 1

        # - Increment episode ID and clear the current episode data
        self._episode_id += 1
        self._clear_episode()

    # -------------------- LOG STEP METRICS --------------------
    def log_step_metrics(self, step: int) -> None:
        # - Return early if there are no pending episodes to log
        if self._pending_episodes == 0:
            return

        # - Compute the mean number of requests per pending episode
        mean_requests = (
            self._pending_requests / self._pending_episodes
        )  # Mean number of requests per episode

        # - Log the mean number of requests per pending episode to MLflow if enabled
        if self.log_to_mlflow:
            mlflow.log_metric(
                (f"num_requests_per_episode/by_training_timesteps/{self.split}"),
                float(mean_requests),
                step=step,
            )

        # - Reset pending requests and episodes for the next logging checkpoint
        self._pending_requests = 0
        self._pending_episodes = 0

    # -------------------- CLEAR EPISODE --------------------
    def _clear_episode(self) -> None:
        if self._arrivals is not None:
            self._arrivals.fill(0)
        if self._stays is not None:
            self._stays.fill(0)
        self._next_day = 0

    # -------------------- FINALIZE TRAINING --------------------
    def finalize_training(self) -> None:
        # - Clear any partial episode and reset pending requests and episodes
        # Do not mix a partial episode with the completed ones.
        self._clear_episode()
        # Timesteps are logged by the callbacks.
        # Do not invent a timestep for any remaining data here.
        self._pending_requests = 0
        self._pending_episodes = 0

        # - Return early if there are no completed episodes to log to MLflow
        if not self.log_to_mlflow or self._completed_episodes == 0:
            return

        assert self._arrivals_sum is not None
        assert self._stays_sum is not None

        # - Log the mean number of arrivals and stays per completed episode to MLflow
        for axis, totals in (
            ("by_request_day", self._arrivals_sum),
            ("by_stay_day", self._stays_sum),
        ):
            for day, total in enumerate(totals):
                mlflow.log_metric(
                    (f"num_requests/{axis}/mean_completed_episodes/{self.split}"),
                    float(total / self._completed_episodes),
                    step=day + 1,
                )
