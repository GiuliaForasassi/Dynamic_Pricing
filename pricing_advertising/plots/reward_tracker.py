from __future__ import annotations

import csv
from io import StringIO
from math import isfinite

import mlflow
import numpy as np


# ----------------------- REWARD TRACKER -----------------------
# Reward tracker for logging and analyzing episode rewards in a simulation.
# Tracks 3 different metrics:
# 1. Total return of the episode
# 2. Mean daily reward of the episode
# 3. Mean daily reward across all completed episodes
class RewardTracker:
    def __init__(self, split: str, log_to_mlflow: bool = True) -> None:
        self.split = split
        self.log_to_mlflow = log_to_mlflow

        self._simulation_days: int | None = None
        self._episode_rewards: list[float] = []
        self._episode_id = 0

        self._day_reward_sum: np.ndarray | None = None
        self._completed_episodes = 0

        self._pending_return_sum = 0.0
        self._pending_episodes = 0

    # -------------------- RECORD DAY --------------------
    def record_day(
        self,
        *,
        day_index: int,
        simulation_days: int,
        reward: float,
    ) -> None:
        # - Validate input parameters
        if simulation_days <= 0:
            raise ValueError("Simulation days must be positive.")

        if self._simulation_days is not None and simulation_days != self._simulation_days:
            raise ValueError("Simulation duration changed: use a separate tracker.")

        if day_index != len(self._episode_rewards) or not 0 <= day_index < simulation_days:
            raise ValueError("Missing, duplicate, or out-of-sequence day.")

        value = float(reward)  # Convert reward to float
        if not isfinite(value):
            raise ValueError("Reward must be finite.")

        # - Initialize arrays if this is the first day of the simulation
        if self._simulation_days is None:
            self._simulation_days = simulation_days
            self._day_reward_sum = np.zeros(
                simulation_days, dtype=float
            )  # Sum of daily rewards across completed episodes

        self._episode_rewards.append(
            value
        )  # Append the reward for the current day to the episode rewards list

    # -------------------- END EPISODE --------------------
    def end_episode(self) -> None:
        # - Return early if there are no episode rewards to process
        if not self._episode_rewards:
            return

        if len(self._episode_rewards) != self._simulation_days:
            raise ValueError("Cannot close an incomplete episode.")

        assert self._day_reward_sum is not None

        # - Convert episode rewards to a NumPy array for efficient computation
        rewards = np.asarray(self._episode_rewards, dtype=float)
        # - Compute the total return for the episode
        episode_return = float(rewards.sum())

        # - Log the total return for the episode to MLflow if enabled
        if self.log_to_mlflow:
            buf = StringIO()
            writer = csv.writer(buf)
            writer.writerow(["day", "reward"])
            writer.writerows(enumerate(self._episode_rewards, start=1))
            mlflow.log_text(
                buf.getvalue(),
                f"episodes/{self.split}/episode_{self._episode_id:05d}_reward.csv",
            )

        # - Update cumulative statistics for the episode
        self._day_reward_sum += rewards
        self._completed_episodes += 1

        self._pending_return_sum += episode_return
        self._pending_episodes += 1

        # - Clear the episode rewards and increment the episode ID for the next episode
        self._episode_rewards.clear()
        self._episode_id += 1

    # -------------------- LOG STEP METRICS --------------------
    def log_step_metrics(self, step: int) -> None:
        if self._pending_episodes == 0:
            return

        assert self._simulation_days is not None

        # - Compute the mean episode return and mean daily reward based on pending episodes
        mean_episode_return = self._pending_return_sum / self._pending_episodes
        mean_daily_reward = mean_episode_return / self._simulation_days

        # - Log the computed mean metrics to MLflow if enabled
        if self.log_to_mlflow:
            mlflow.log_metric(
                f"episode_return/by_training_timesteps/{self.split}",
                float(mean_episode_return),
                step=step,
            )
            mlflow.log_metric(
                f"mean_daily_reward/by_training_timesteps/{self.split}",
                float(mean_daily_reward),
                step=step,
            )

        # - Reset the pending return sum and pending episodes count after logging
        self._pending_return_sum = 0.0
        self._pending_episodes = 0

    # -------------------- FINALIZE TRAINING --------------------
    def finalize_training(self) -> None:
        # - Finalize the training by discarding incomplete episodes and logging the mean rewards for completed episodes
        # Discard the incomplete episode: it should not be averaged with the completed ones.
        self._episode_rewards.clear()

        # - Reset the pending return sum and pending episodes count to ensure no leftover data affects future logging
        self._pending_return_sum = 0.0
        self._pending_episodes = 0

        if not self.log_to_mlflow or self._completed_episodes == 0:
            return

        assert self._day_reward_sum is not None

        # - Log the mean daily rewards for each request day across all completed episodes
        for day_index, total in enumerate(self._day_reward_sum):
            mlflow.log_metric(
                (f"reward/by_request_day/mean_completed_episodes/{self.split}"),
                float(total / self._completed_episodes),
                step=day_index + 1,
            )
