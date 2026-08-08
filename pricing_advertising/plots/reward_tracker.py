from __future__ import annotations

import csv
from io import StringIO

import mlflow
import numpy as np


class RewardTracker:
    """Accumulates per-day reward and logs it to mlflow, per training split.

    Two metrics are produced (keys suffixed with `split`, e.g. "train"/"val"):
      - "reward/by_training_timestep/{split}": mean reward over an episode (or, for eval,
        over a whole eval round), logged against a caller-supplied `step` (typically
        `num_timesteps`, so train/val curves share the same x-axis and can be
        overlaid/compared in the MLflow UI). See `end_episode`/`log_step_metric`.
      - "reward/by_day/{split}": one point per simulation day, averaged
        across all episodes seen so far, logged once `finalize_training` is
        called (i.e. at the end of training).

    A CSV artifact (`day,reward`) is also logged per episode.
    """

    def __init__(self, split: str, log_to_mlflow: bool = True) -> None:
        self.split = split  # train or val
        self.log_to_mlflow = log_to_mlflow

        self._episode_id: int = 0
        self._episode_reward: list[float] = []

        self._day_reward_sum: np.ndarray | None = None
        self._day_reward_count: np.ndarray | None = None

    def record_day(self, reward: float) -> None:
        self._episode_reward.append(float(reward))

    def end_episode(self) -> float | None:
        """Flush the current episode: write its CSV artifact, accumulate its
        per-day values into the running by-day averages, and return the
        episode's mean reward (or None if no data was recorded).
        """
        if not self._episode_reward:
            return None

        # Calculates the episode mean
        episode_reward = np.asarray(self._episode_reward, dtype=float)
        mean_reward = float(episode_reward.mean())

        # Save a CSV artifact on MLflow
        if self.log_to_mlflow:
            buf = StringIO()
            writer = csv.writer(buf)
            writer.writerow(["day", "reward"])
            writer.writerows(enumerate(self._episode_reward))
            mlflow.log_text(
                buf.getvalue(),
                f"episodes/{self.split}/episode_{self._episode_id:05d}_reward.csv",
            )

        # Collect daily data
        if self._day_reward_sum is None or self._day_reward_count is None:
            self._day_reward_sum = np.zeros_like(episode_reward)
            self._day_reward_count = np.zeros_like(episode_reward, dtype=int)
        # If episodes have different durations we use padding
        if self._day_reward_sum.shape != episode_reward.shape:
            padding = np.zeros(len(self._day_reward_sum) - len(episode_reward))
            counts_considering_padding = np.concatenate(
                [np.ones_like(episode_reward), np.zeros_like(padding)]
            ).astype(np.int64)
            episode_reward = np.concatenate([episode_reward, padding])
            self._day_reward_count += counts_considering_padding
        else:
            self._day_reward_count += 1
        self._day_reward_sum += episode_reward

        self._episode_reward = []
        self._episode_id += 1

        return mean_reward

    def log_step_metric(self, value: float, step: int) -> None:
        """Log a single scalar (e.g. mean episode reward) against `step`
        (typically `num_timesteps`), so train/val curves are directly
        comparable in the MLflow UI.
        """
        if self.log_to_mlflow:
            mlflow.log_metric(f"reward/by_training_timestep/{self.split}", value, step=step)

    def finalize_training(self) -> None:
        # Closes the latest episode if there is one still in progress
        if self._episode_reward:
            self.end_episode()

        day_reward_sum, day_reward_count = self._day_reward_sum, self._day_reward_count
        if not self.log_to_mlflow or day_reward_sum is None or day_reward_count is None:
            return

        # Calculates the avg reward taking into account all episodes and send this data to MLflow
        for day in range(len(day_reward_sum)):
            mlflow.log_metric(
                f"reward/by_day/{self.split}",
                float(day_reward_sum[day] / day_reward_count[day]),
                step=day,
            )
