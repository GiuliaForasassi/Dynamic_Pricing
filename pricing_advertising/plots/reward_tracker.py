from __future__ import annotations

import csv
from io import StringIO
from math import isfinite

import mlflow
import numpy as np


class RewardTracker:
    """Reward per giorno di richiesta, su episodi completi.

    Registra:
    - ritorno non scontato dell'episodio;
    - reward medio giornaliero dell'episodio;
    - reward per giorno, mediato sugli episodi completi della run.

    In evaluation i riepiloghi temporali vengono mediati sugli episodi
    del singolo checkpoint.
    """

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

    def record_day(
        self,
        *,
        day_index: int,
        simulation_days: int,
        reward: float,
    ) -> None:
        if simulation_days <= 0:
            raise ValueError("Durata della simulazione non positiva.")

        if self._simulation_days is not None and simulation_days != self._simulation_days:
            raise ValueError("Durata cambiata: usare un tracker separato.")

        if day_index != len(self._episode_rewards) or not 0 <= day_index < simulation_days:
            raise ValueError("Giorno mancante, duplicato o fuori sequenza.")

        value = float(reward)
        if not isfinite(value):
            raise ValueError("Il reward deve essere finito.")

        if self._simulation_days is None:
            self._simulation_days = simulation_days
            self._day_reward_sum = np.zeros(simulation_days, dtype=float)

        self._episode_rewards.append(value)

    def end_episode(self) -> None:
        if not self._episode_rewards:
            return

        if len(self._episode_rewards) != self._simulation_days:
            raise ValueError("Impossibile chiudere un episodio incompleto.")

        assert self._day_reward_sum is not None

        rewards = np.asarray(self._episode_rewards, dtype=float)
        episode_return = float(rewards.sum())

        if self.log_to_mlflow:
            buf = StringIO()
            writer = csv.writer(buf)
            writer.writerow(["day", "reward"])
            writer.writerows(enumerate(self._episode_rewards, start=1))
            mlflow.log_text(
                buf.getvalue(),
                f"episodes/{self.split}/episode_{self._episode_id:05d}_reward.csv",
            )

        self._day_reward_sum += rewards
        self._completed_episodes += 1

        self._pending_return_sum += episode_return
        self._pending_episodes += 1

        self._episode_rewards.clear()
        self._episode_id += 1

    def log_step_metrics(self, step: int) -> None:
        if self._pending_episodes == 0:
            return

        assert self._simulation_days is not None

        mean_episode_return = self._pending_return_sum / self._pending_episodes
        mean_daily_reward = mean_episode_return / self._simulation_days

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

        self._pending_return_sum = 0.0
        self._pending_episodes = 0

    def finalize_training(self) -> None:
        # Scartiamo l'episodio incompleto: non va mediato con quelli completi.
        self._episode_rewards.clear()

        # I punti temporali sono registrati dai callback, non qui.
        self._pending_return_sum = 0.0
        self._pending_episodes = 0

        if not self.log_to_mlflow or self._completed_episodes == 0:
            return

        assert self._day_reward_sum is not None

        for day_index, total in enumerate(self._day_reward_sum):
            mlflow.log_metric(
                (f"reward/by_request_day/mean_completed_episodes/{self.split}"),
                float(total / self._completed_episodes),
                step=day_index + 1,
            )
