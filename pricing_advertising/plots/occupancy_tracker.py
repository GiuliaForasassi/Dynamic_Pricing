from __future__ import annotations

import csv
from io import StringIO

import mlflow
import numpy as np


class OccupancyTracker:
    """Accumulates per-day occupancy and logs it to mlflow, per training split.

    Two metrics are produced (keys suffixed with `split`, e.g. "train"/"val"):
      - "occupancy/{split}": mean occupancy over an episode (or, for eval, over
        a whole eval round), logged against a caller-supplied `step` (typically
        `num_timesteps`, so train/val curves share the same x-axis and can be
        overlaid/compared in the MLflow UI). See `end_episode`/`log_step_metric`.
      - "occupancy/by_day/{split}": one point per simulation day, averaged
        across all episodes seen so far, logged once `finalize_training` is
        called (i.e. at the end of training).

    A CSV artifact (`day,occupancy_pct`) is also logged per episode.
    """

    def __init__(self, split: str, log_to_mlflow: bool = True) -> None:
        self.split = split  # train o val
        self.log_to_mlflow = log_to_mlflow

        self._episode_id: int = 0
        self._episode_occ: list[float] = []

        self._day_occ_sum: np.ndarray | None = None
        self._day_occ_count: np.ndarray | None = None

    def record_day(self, occupancy_pct: float) -> None:
        self._episode_occ.append(float(occupancy_pct))

    def end_episode(self) -> float | None:
        """Flush the current episode: write its CSV artifact, accumulate its
        per-day values into the running by-day averages, and return the
        episode's mean occupancy (or None if no data was recorded).
        """
        if not self._episode_occ:
            return None

        # Calculates the episode mean
        episode_occ = np.asarray(self._episode_occ, dtype=float)
        mean_occ = float(episode_occ.mean())

        # Save an CSV artifact on MLflow
        if self.log_to_mlflow:
            buf = StringIO()
            writer = csv.writer(buf)
            writer.writerow(["day", "occupancy_pct"])
            writer.writerows(enumerate(self._episode_occ))
            mlflow.log_text(
                buf.getvalue(),
                f"episodes/{self.split}/episode_{self._episode_id:05d}_occupancy.csv",
            )

        # Collect daily data
        if self._day_occ_sum is None or self._day_occ_count is None:
            self._day_occ_sum = np.zeros_like(episode_occ)
            self._day_occ_count = np.zeros_like(episode_occ, dtype=int)
        # If episodes have different durantions we use padding
        if self._day_occ_sum.shape != episode_occ.shape:
            padding = np.zeros(len(self._day_occ_sum) - len(episode_occ))
            counts_considering_padding = np.concat(
                [np.ones_like(episode_occ), np.zeros_like(padding)]
            ).astype(np.int64)
            episode_occ = np.concat([episode_occ, padding])
            self._day_occ_count += counts_considering_padding
        else:
            self._day_occ_count += 1
        self._day_occ_sum += episode_occ

        self._episode_occ = []
        self._episode_id += 1

        return mean_occ

    def log_step_metric(self, value: float, step: int) -> None:
        """Log a single scalar (e.g. mean episode occupancy) against `step`
        (typically `num_timesteps`), so train/val curves are directly
        comparable in the MLflow UI.
        """
        if self.log_to_mlflow:
            mlflow.log_metric(f"occupancy/by_training_timesteps/{self.split}", value, step=step)

    def finalize_training(self) -> None:
        # Closes the latest episode if there is one still in progress
        if self._episode_occ:
            self.end_episode()

        day_occ_sum, day_occ_count = self._day_occ_sum, self._day_occ_count
        if not self.log_to_mlflow or day_occ_sum is None or day_occ_count is None:
            return

        # Calculates the avg occupancy taking into account all episodes and send this data to MLflow
        for day in range(len(day_occ_sum)):
            mlflow.log_metric(
                f"occupancy/by_day/{self.split}",
                float(day_occ_sum[day] / day_occ_count[day]),
                step=day,
            )
