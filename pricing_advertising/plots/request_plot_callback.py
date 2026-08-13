from __future__ import annotations

import tempfile
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RequestPlotCallback(BaseCallback):
    """Generates request count plots showing number of requests by request date.

    For selected request dates (t_request), generates plots showing the count of
    requests for each occupancy day (t_reservation) starting from that
    request date through the end of simulation. Plots are generated at evaluation
    frequency and saved as MLflow artifacts.
    """

    def __init__(
        self,
        eval_freq: int = 500,
        n_envs: int = 4,
        log_to_mlflow: bool = True,
        track_env_idx: int = 0,
    ) -> None:
        """Initialize the request plot callback.

        Args:
            eval_freq: Evaluation frequency in timesteps.
            n_envs: Number of parallel environments.
            log_to_mlflow: Whether to log plots to MLflow.
            track_env_idx: Environment index to track.
        """
        super().__init__()
        self.eval_freq = eval_freq
        self.n_envs = n_envs
        self.log_to_mlflow = log_to_mlflow
        self.track_env_idx = track_env_idx

        # Generate plots every n_envs * eval_freq timesteps
        self.plot_freq = n_envs * eval_freq

        # Request dates to generate plots for (will be set by user)
        self.request_dates: list[date] = []

        # Simulation parameters (set on first step)
        self._start_simulation: date | None = None
        self._end_simulation: date | None = None
        self._simulation_days: int | None = None
        self._initialized = False

        # Track request count by (t_request, t_reservation)
        # Structure: dict[t_request] -> dict[t_reservation] -> count
        self._requests_by_request_date: dict[date, dict[date, int]] = {}

        self._plot_count = 0

    def _get_attr(self, name: str, idx: int) -> Any:  # noqa: ANN401
        """Get attribute from environment at index idx."""
        return self.training_env.get_attr(name, indices=idx)[0]

    def _initialize(self) -> None:
        """Initialize parameters from environment."""
        if self._initialized:
            return

        env_idx = self.track_env_idx
        self._start_simulation = self._get_attr("start_simulation", env_idx)
        self._end_simulation = self._get_attr("end_simulation", env_idx)

        if self._start_simulation and self._end_simulation:
            self._simulation_days = (self._end_simulation - self._start_simulation).days + 1
            self._initialized = True

    def set_request_dates(self, request_dates: list[date] | list[tuple[int, int, int]]) -> None:
        """Set the request dates to generate plots for.

        Args:
            request_dates: List of dates or list of (year, month, day) tuples.
        """
        self.request_dates = []
        for req_date in request_dates:
            if isinstance(req_date, tuple):
                self.request_dates.append(date(*req_date))
            else:
                self.request_dates.append(req_date)

    def record_requests(
        self,
        t_request: date,
        t_reservation: date,
        num_days: int,
    ) -> None:
        """Record a request and its occupancy days.

        Args:
            t_request: The request date.
            t_reservation: The check-in date (occupancy start).
            num_days: Number of days for the stay.
        """
        if t_request not in self._requests_by_request_date:
            self._requests_by_request_date[t_request] = {}

        # Increment count for each day of the stay
        for day_offset in range(num_days):
            current_t_reservation = t_reservation + timedelta(days=day_offset)

            if current_t_reservation not in self._requests_by_request_date[t_request]:
                self._requests_by_request_date[t_request][current_t_reservation] = 0

            self._requests_by_request_date[t_request][current_t_reservation] += 1

    def _generate_plot_for_request_date(self, t_request: date, eval_step: int) -> bytes:
        """Generate a request count plot for a specific request date.

        Args:
            t_request: The request date to generate plot for.
            eval_step: The evaluation step count.

        Returns:
            PNG image as bytes.
        """
        if (
            self._start_simulation is None
            or self._end_simulation is None
            or t_request not in self._requests_by_request_date
        ):
            raise ValueError("Invalid state for plot generation")

        # Collect request counts for each reservation day
        reservation_days = []
        request_counts = []

        # Generate days from t_request to end_simulation
        current_day = t_request
        while current_day <= self._end_simulation:
            if current_day in self._requests_by_request_date[t_request]:
                count = self._requests_by_request_date[t_request][current_day]
            else:
                count = 0

            reservation_days.append(current_day)
            request_counts.append(count)
            current_day += timedelta(days=1)

        # Convert days to day-of-year indices for plotting (starting from simulation start)
        day_indices = [(day - self._start_simulation).days for day in reservation_days]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(day_indices, request_counts, marker="o", linestyle="-", linewidth=1.5, markersize=3)
        ax.set_xlabel("Day of Year")
        ax.set_ylabel("Request Count")
        ax.set_title(f"Request Count - Request Date: {t_request.strftime('%Y-%m-%d')}")
        ax.grid(True, alpha=0.3)

        # Convert to bytes
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close(fig)

        return image_bytes

    def _generate_all_plots(self) -> None:
        """Generate plots for all request dates and log to MLflow."""
        if not self._initialized or self._start_simulation is None:
            return

        eval_step = self.num_timesteps

        for t_request in self.request_dates:
            # Skip if no data for this request date
            if t_request not in self._requests_by_request_date:
                continue

            try:
                # Generate plot
                plot_bytes = self._generate_plot_for_request_date(t_request, eval_step)

                # Create artifact path with request date as folder and step count in filename
                request_date_str = t_request.strftime("%Y-%m-%d")
                filename = f"{eval_step}_eval_timesteps.png"
                artifact_path = f"request_plots/{request_date_str}"

                if self.log_to_mlflow:
                    # Create temporary directory and file with the desired filename
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_file_path = Path(tmpdir) / filename
                        tmp_file_path.write_bytes(plot_bytes)
                        mlflow.log_artifact(str(tmp_file_path), artifact_path=artifact_path)

                self._plot_count += 1
            except Exception as e:
                print(f"Error generating request plot for request date {t_request}: {e}")

    def _reset_metrics(self) -> None:
        """Clear request metrics for the next evaluation period."""
        self._requests_by_request_date.clear()

    def _on_step(self) -> bool:
        """Called after each environment step."""
        if not self._initialized:
            self._initialize()

        n_envs = int(self.training_env.num_envs)
        env_idx = self.track_env_idx
        if env_idx < 0 or env_idx >= n_envs:
            return True

        # Get the final requests from the current step
        try:
            final_requests = self._get_attr("last_final_requests", env_idx)
        except (AttributeError, IndexError):
            final_requests = None

        # Process each request
        if final_requests is not None and len(final_requests) > 0:
            for req in final_requests:
                self.record_requests(req.t_request, req.t_reservation, req.days_stay)

        # Generate plots at the evaluation frequency
        if self.num_timesteps % self.plot_freq == 0 and self.num_timesteps > 0:
            self._generate_all_plots()
            self._reset_metrics()

        return True

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Generate final plots
        if self._initialized:
            self._generate_all_plots()
