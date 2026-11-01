import csv
from io import StringIO
from math import isfinite

import mlflow
import numpy as np


class CancellationTracker:
    def __init__(self, split: str, log_to_mlflow: bool = True) -> None:
        self.split = split
        self.log_to_mlflow = log_to_mlflow
        self._simulation_days: int | None = None
        self._episode_counts: list[int] = []
        self._day_sum: np.ndarray | None = None
        self._completed_episodes = 0
        self._pending_mean_sum = 0.0
        self._pending_episodes = 0

    def record_day(self, *, day_index: int, simulation_days: int, num_cancelled: int) -> None:
        if simulation_days <= 0:
            raise ValueError("Simulation days must be positive.")
        if self._simulation_days is not None and simulation_days != self._simulation_days:
            raise ValueError("Duration changed: use a separate tracker.")
        if day_index != len(self._episode_counts) or not 0 <= day_index < simulation_days:
            raise ValueError("Missing, duplicate, or out-of-sequence day.")
        value = float(num_cancelled)
        if not isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError("Cancellation count must be a finite non-negative integer.")
        if self._simulation_days is None:
            self._simulation_days = simulation_days
            self._day_sum = np.zeros(simulation_days, dtype=float)
        self._episode_counts.append(int(value))

    def end_episode(self) -> None:
        if not self._episode_counts:
            return
        if len(self._episode_counts) != self._simulation_days:
            raise ValueError("Cannot close an incomplete episode.")
        counts = np.asarray(self._episode_counts, dtype=float)
        if self.log_to_mlflow:
            buffer = StringIO()
            writer = csv.writer(buffer)
            writer.writerow(["day", "cancellation_amount"])
            writer.writerows(enumerate(self._episode_counts, start=1))
            mlflow.log_text(
                buffer.getvalue(),
                f"episodes/{self.split}/episode_{self._completed_episodes:05d}_cancellation.csv",
            )
        assert self._day_sum is not None
        self._day_sum += counts
        self._completed_episodes += 1
        self._pending_mean_sum += float(counts.mean())
        self._pending_episodes += 1
        self._episode_counts.clear()

    def log_step_metrics(self, step: int) -> None:
        if not self._pending_episodes:
            return
        if self.log_to_mlflow:
            mlflow.log_metric(
                f"cancellation_amount/by_training_timesteps/{self.split}",
                self._pending_mean_sum / self._pending_episodes,
                step=step,
            )
        self._pending_mean_sum = 0.0
        self._pending_episodes = 0

    def finalize_training(self) -> None:
        self._episode_counts.clear()
        self._pending_mean_sum = 0.0
        self._pending_episodes = 0
        if not self.log_to_mlflow or not self._completed_episodes:
            return
        assert self._day_sum is not None
        for day_index, total in enumerate(self._day_sum):
            mlflow.log_metric(
                f"cancellation_amount/by_cancellation_day/mean_completed_episodes/{self.split}",
                float(total / self._completed_episodes),
                step=day_index + 1,
            )
