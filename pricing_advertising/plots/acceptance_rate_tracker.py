from __future__ import annotations

import mlflow
import numpy as np


class AcceptanceRateTracker:
    """Accumulates per-day acceptance rate and logs it to mlflow, per training split.

    Two metrics are produced (keys suffixed with `split`, e.g. "train"/"val"):
      - "acceptance_rate/by_training_timestep/{split}": mean acceptance rate over an episode
        (or, for eval, over a whole eval round), logged against a caller-supplied `step`
        (typically `num_timesteps`), so train/val curves share the same x-axis and can be
        overlaid/compared in the MLflow UI. See `end_episode`/`log_step_metric`.
      - "acceptance_rate/by_reservation_day/{split}": one point per simulation day (0-365),
        averaged across all episodes seen so far, logged once `finalize_training` is
        called (i.e. at the end of training).

    Acceptance rate is calculated as: accepted_count / (accepted_count + rejected_count)
    for each day, aggregated across all room categories.
    """

    def __init__(self, split: str, log_to_mlflow: bool = True) -> None:
        self.split = split  # train or val
        self.log_to_mlflow = log_to_mlflow

        self._episode_id: int = 0
        self._episode_acceptance_rate: list[float] = []

        self._day_acceptance_rate_sum: np.ndarray | None = None
        self._day_acceptance_rate_count: np.ndarray | None = None

    def record_day(self, acceptance_rate: float) -> None:
        """Record acceptance rate for a single day.

        Args:
            acceptance_rate: Acceptance rate (0.0 to 1.0) for this day.
        """
        self._episode_acceptance_rate.append(float(acceptance_rate))

    def end_episode(self) -> float | None:
        """Flush the current episode: accumulate per-day values into running averages,
        and return the episode's mean acceptance rate (or None if no data was recorded).
        """
        if not self._episode_acceptance_rate:
            return None

        episode_rates = np.asarray(self._episode_acceptance_rate, dtype=float)
        mean_rate = float(episode_rates.mean())

        # Accumulate into across-episode statistics
        if self._day_acceptance_rate_sum is None or self._day_acceptance_rate_count is None:
            self._day_acceptance_rate_sum = np.zeros_like(episode_rates, dtype=float)
            self._day_acceptance_rate_count = np.zeros_like(episode_rates, dtype=int)
        # If episodes have different durations we use padding
        if self._day_acceptance_rate_sum.shape != episode_rates.shape:
            padding = np.zeros(len(self._day_acceptance_rate_sum) - len(episode_rates))
            counts_considering_padding = np.concatenate(
                [np.ones_like(episode_rates), np.zeros_like(padding)]
            ).astype(np.int64)
            episode_rates = np.concatenate([episode_rates, padding])
            self._day_acceptance_rate_count += counts_considering_padding
        else:
            self._day_acceptance_rate_count += 1
        self._day_acceptance_rate_sum += episode_rates

        self._episode_acceptance_rate = []
        self._episode_id += 1

        return mean_rate

    def log_step_metric(self, value: float, step: int) -> None:
        """Log a single scalar (e.g. mean episode acceptance rate) against `step`
        (typically `num_timesteps`), so train/val curves are directly
        comparable in the MLflow UI.
        """
        if self.log_to_mlflow:
            mlflow.log_metric(
                f"acceptance_rate/by_training_timestep/{self.split}", value, step=step
            )

    def finalize_training(self) -> None:
        """Finalize training: flush any remaining episode data and log per-day
        averages to mlflow.
        """
        # Closes the latest episode if there is one still in progress
        if self._episode_acceptance_rate:
            self.end_episode()

        if (
            not self.log_to_mlflow
            or self._day_acceptance_rate_sum is None
            or self._day_acceptance_rate_count is None
        ):
            return

        # Calculates the avg acceptance rate taking into account all episodes and send this data to MLflow
        for day in range(len(self._day_acceptance_rate_sum)):
            if self._day_acceptance_rate_count[day] > 0:
                mlflow.log_metric(
                    f"acceptance_rate/by_reservation_day/{self.split}",
                    float(
                        self._day_acceptance_rate_sum[day] / self._day_acceptance_rate_count[day]
                    ),
                    step=day,
                )
