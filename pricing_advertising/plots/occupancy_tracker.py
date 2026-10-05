from __future__ import annotations

import csv
from io import StringIO
from math import isfinite

import mlflow
import numpy as np


# --------------- OCCUPANCY TRACKER ------------------
# It tracks the occupancy percentage of each day across complete episodes.
# The value recorded each day represents the occupancy rate for the current night,
# that is, the percentage of rooms occupied that night.
class OccupancyTracker:
    def __init__(self, split: str, log_to_mlflow: bool = True) -> None:
        self.split = split
        self.log_to_mlflow = log_to_mlflow

        self._simulation_days: int | None = None
        self._episode_occ: list[float] = []  # Occupancy percentages for the current episode
        self._episode_id = 0

        self._day_occ_sum: np.ndarray | None = (
            None  # Sum of occupancy percentages for each day across completed episodes
        )
        self._completed_episodes = 0

        # Pending metrics for the current step before logging
        self._pending_mean_sum = 0.0
        self._pending_episodes = 0

    def record_day(
        self,
        *,
        day_index: int,
        simulation_days: int,
        occupancy_pct: float,
    ) -> None:
        # Check validity of the input parameters
        if simulation_days <= 0:
            raise ValueError("Simulation days must be positive.")

        if self._simulation_days is not None and simulation_days != self._simulation_days:
            raise ValueError("Duration changed: use a separate tracker.")

        if day_index != len(self._episode_occ) or not 0 <= day_index < simulation_days:
            raise ValueError("Missing, duplicate, or out-of-sequence day.")

        value = float(occupancy_pct)
        if not isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError("Invalid occupancy: must be finite and between 0 and 100.")

        # Initialize the simulation days and day occupancy sum
        if self._simulation_days is None:
            self._simulation_days = simulation_days
            self._day_occ_sum = np.zeros(simulation_days, dtype=float)

        self._episode_occ.append(value)

    # --------------- END OF EPISODE ------------------
    # It's called at the end of an episode (when done is true) to finalize and log the occupancy data.
    def end_episode(self) -> None:
        # Checks
        if not self._episode_occ:
            return

        if len(self._episode_occ) != self._simulation_days:
            raise ValueError("Cannot close an incomplete episode.")

        assert self._day_occ_sum is not None

        # Convert the episode occupancy list to a NumPy array for efficient computation
        occupancy = np.asarray(self._episode_occ, dtype=float)
        # Compute the mean occupancy for the episode
        mean_occupancy = float(occupancy.mean())

        # Log the episode occupancy to MLflow if enabled
        if self.log_to_mlflow:
            buf = StringIO()  # Create an in-memory buffer to store the CSV data
            writer = csv.writer(
                buf
            )  # Create a CSV writer to write the occupancy data into the buffer
            writer.writerow(["day", "occupancy_pct"])  # Write the header row to the CSV buffer
            writer.writerows(
                enumerate(self._episode_occ, start=1)
            )  # Write the occupancy data rows to the CSV buffer
            mlflow.log_text(
                buf.getvalue(),
                (f"episodes/{self.split}/episode_{self._episode_id:05d}_occupancy.csv"),
            )  # Log the CSV data to MLflow

        # Update the cumulative day occupancy sum and episode counters
        self._day_occ_sum += occupancy
        self._completed_episodes += 1

        self._pending_mean_sum += mean_occupancy
        self._pending_episodes += 1

        self._episode_occ.clear()
        self._episode_id += 1

    # ----------------- LOG STEP METRICS ------------------
    # Logs the mean occupancy for the pending episodes to MLflow at the given training step.
    def log_step_metrics(self, step: int) -> None:
        if self._pending_episodes == 0:
            return
        # Compute the mean occupancy for the pending episodes
        mean_occupancy = self._pending_mean_sum / self._pending_episodes
        # Log the mean occupancy for the pending episodes to MLflow if enabled
        if self.log_to_mlflow:
            mlflow.log_metric(
                f"occupancy_pct/by_training_timesteps/{self.split}",
                float(mean_occupancy),
                step=step,
            )
        # Reset the pending mean sum and episode counters after logging
        self._pending_mean_sum = 0.0
        self._pending_episodes = 0

    # ----------------- FINALIZE TRAINING ------------------
    # Finalizes the training by logging the mean occupancy for each day across all completed episodes to MLflow.
    def finalize_training(self) -> None:
        # Clear any remaining episode data and reset pending metrics before finalizing training.
        self._episode_occ.clear()
        self._pending_mean_sum = 0.0
        self._pending_episodes = 0

        # If there are no completed episodes or MLflow logging is disabled, exit early.
        if not self.log_to_mlflow or self._completed_episodes == 0:
            return

        # Ensure that the cumulative day occupancy sum is available before logging.
        assert self._day_occ_sum is not None

        # Log the mean occupancy for each day across all completed episodes to MLflow.
        for day_index, total in enumerate(self._day_occ_sum):
            mlflow.log_metric(
                (f"occupancy_pct/by_stay_day/mean_completed_episodes/{self.split}"),
                float(total / self._completed_episodes),
                step=day_index + 1,
            )
