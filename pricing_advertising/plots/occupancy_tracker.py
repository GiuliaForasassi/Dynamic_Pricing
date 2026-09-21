from __future__ import annotations

import csv
from io import StringIO
from math import isfinite

import mlflow
import numpy as np


class OccupancyTracker:
    """Occupazione percentuale della notte corrente, su episodi completi.

    La durata deve essere costante per il tracker.
    La media temporale rappresenta l'utilizzo delle camere-notte
    quando la capacità dell'hotel è costante.
    """

    def __init__(self, split: str, log_to_mlflow: bool = True) -> None:
        self.split = split
        self.log_to_mlflow = log_to_mlflow

        self._simulation_days: int | None = None
        self._episode_occ: list[float] = []
        self._episode_id = 0

        self._day_occ_sum: np.ndarray | None = None
        self._completed_episodes = 0

        self._pending_mean_sum = 0.0
        self._pending_episodes = 0

    def record_day(
        self,
        *,
        day_index: int,
        simulation_days: int,
        occupancy_pct: float,
    ) -> None:
        if simulation_days <= 0:
            raise ValueError("Durata della simulazione non positiva.")

        if self._simulation_days is not None and simulation_days != self._simulation_days:
            raise ValueError("Durata cambiata: usare un tracker separato.")

        if day_index != len(self._episode_occ) or not 0 <= day_index < simulation_days:
            raise ValueError("Giorno mancante, duplicato o fuori sequenza.")

        value = float(occupancy_pct)
        if not isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError("Occupancy non valida: deve essere finita e tra 0 e 100.")

        if self._simulation_days is None:
            self._simulation_days = simulation_days
            self._day_occ_sum = np.zeros(simulation_days, dtype=float)

        self._episode_occ.append(value)

    def end_episode(self) -> None:
        if not self._episode_occ:
            return

        if len(self._episode_occ) != self._simulation_days:
            raise ValueError("Impossibile chiudere un episodio incompleto.")

        assert self._day_occ_sum is not None

        occupancy = np.asarray(self._episode_occ, dtype=float)
        mean_occupancy = float(occupancy.mean())

        if self.log_to_mlflow:
            buf = StringIO()
            writer = csv.writer(buf)
            writer.writerow(["day", "occupancy_pct"])
            writer.writerows(enumerate(self._episode_occ, start=1))
            mlflow.log_text(
                buf.getvalue(),
                (f"episodes/{self.split}/episode_{self._episode_id:05d}_occupancy.csv"),
            )

        self._day_occ_sum += occupancy
        self._completed_episodes += 1

        self._pending_mean_sum += mean_occupancy
        self._pending_episodes += 1

        self._episode_occ.clear()
        self._episode_id += 1

    def log_step_metrics(self, step: int) -> None:
        if self._pending_episodes == 0:
            return

        mean_occupancy = self._pending_mean_sum / self._pending_episodes

        if self.log_to_mlflow:
            mlflow.log_metric(
                f"occupancy_pct/by_training_timesteps/{self.split}",
                float(mean_occupancy),
                step=step,
            )

        self._pending_mean_sum = 0.0
        self._pending_episodes = 0

    def finalize_training(self) -> None:
        # Non includiamo l'episodio parziale nelle medie.
        self._episode_occ.clear()
        self._pending_mean_sum = 0.0
        self._pending_episodes = 0

        if not self.log_to_mlflow or self._completed_episodes == 0:
            return

        assert self._day_occ_sum is not None

        for day_index, total in enumerate(self._day_occ_sum):
            mlflow.log_metric(
                (f"occupancy_pct/by_stay_day/mean_completed_episodes/{self.split}"),
                float(total / self._completed_episodes),
                step=day_index + 1,
            )
