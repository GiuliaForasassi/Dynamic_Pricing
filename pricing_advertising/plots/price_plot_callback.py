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


class PricePlotCallback(BaseCallback):
    """Generates price plots showing average daily prices by request date.

    For selected request dates (t_request), generates plots showing the average daily
    occupancy price (t_reservation) for all days starting from that request date through
    the end of simulation. Plots are generated at evaluation frequency and saved as
    MLflow artifacts.
    """

    def __init__(
        self,
        eval_freq: int = 500,
        n_envs: int = 4,
        log_to_mlflow: bool = True,
        track_env_idx: int = 0,
    ) -> None:
        """Initialize the price plot callback.

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

        # Track prices by (t_request, t_reservation)
        # Structure: dict[t_request] -> dict[t_reservation] -> list of prices
        self._prices_by_request: dict[date, dict[date, list[float]]] = {}

        # Track count of prices for averaging
        self._price_counts_by_request: dict[date, dict[date, int]] = {}

        self._plot_count = 0

    def _get_attr(self, name: str, idx: int) -> Any:  # noqa: ANN401
        """Get attribute from environment at index idx."""
        return self.training_env.get_attr(name, indices=idx)[0]

    def _env_method(self, method_name: str, *args: Any, idx: int, **kwargs: Any) -> Any:  # noqa: ANN401
        """Call method on environment at index idx."""
        return self.training_env.env_method(method_name, *args, indices=idx, **kwargs)[0]

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
                          E.g., [(2025, 2, 28), (2025, 5, 15), (2025, 8, 1), (2025, 11, 12)]
        """
        self.request_dates = []
        for req_date in request_dates:
            if isinstance(req_date, tuple):
                self.request_dates.append(date(*req_date))
            else:
                self.request_dates.append(req_date)

    def record_prices(
        self,
        t_request: date,
        t_reservation: date,
        daily_prices: np.ndarray,
    ) -> None:
        """Record prices for a set of requests.

        Args:
            t_request: The request date.
            t_reservation: The check-in date (occupancy start).
            daily_prices: Array of prices for each day of stay.
        """
        if t_request not in self._prices_by_request:
            self._prices_by_request[t_request] = {}
            self._price_counts_by_request[t_request] = {}

        # Record prices for each day of the stay
        for day_offset, price in enumerate(daily_prices):
            current_t_reservation = t_reservation + timedelta(days=day_offset)

            if current_t_reservation not in self._prices_by_request[t_request]:
                self._prices_by_request[t_request][current_t_reservation] = []
                self._price_counts_by_request[t_request][current_t_reservation] = 0

            self._prices_by_request[t_request][current_t_reservation].append(float(price))
            self._price_counts_by_request[t_request][current_t_reservation] += 1

    def _generate_plot_for_request_date(self, t_request: date, eval_step: int) -> bytes:
        """Generate a price plot for a specific request date.

        Args:
            t_request: The request date to generate plot for.
            eval_step: The evaluation step count.

        Returns:
            PNG image as bytes.
        """
        if (
            self._start_simulation is None
            or self._end_simulation is None
            or t_request not in self._prices_by_request
        ):
            raise ValueError("Invalid state for plot generation")

        # Collect average prices for each reservation day
        reservation_days = []
        avg_prices = []

        # Generate days from t_request to end_simulation
        current_day = t_request
        while current_day <= self._end_simulation:
            if current_day in self._prices_by_request[t_request]:
                prices = self._prices_by_request[t_request][current_day]
                avg_price = float(np.mean(prices)) if prices else 0.0
            else:
                avg_price = 0.0

            reservation_days.append(current_day)
            avg_prices.append(avg_price)
            current_day += timedelta(days=1)

        # Convert days to day-of-year indices for plotting (starting from simulation start)
        day_indices = [(day - self._start_simulation).days for day in reservation_days]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(day_indices, avg_prices, marker="o", linestyle="-", linewidth=1.5, markersize=3)
        ax.set_xlabel("Day of Year")
        ax.set_ylabel("Average Daily Price")
        ax.set_title(f"Average Daily Price - Request Date: {t_request.strftime('%Y-%m-%d')}")
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
            if t_request not in self._prices_by_request:
                continue

            try:
                # Generate plot
                plot_bytes = self._generate_plot_for_request_date(t_request, eval_step)

                # Create artifact path with request date as folder and step count in filename
                request_date_str = t_request.strftime("%Y-%m-%d")
                filename = f"{eval_step}_eval_timesteps.png"
                artifact_path = f"price_plots/{request_date_str}"

                if self.log_to_mlflow:
                    # Create temporary directory and file with the desired filename
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_file_path = Path(tmpdir) / filename
                        tmp_file_path.write_bytes(plot_bytes)
                        mlflow.log_artifact(str(tmp_file_path), artifact_path=artifact_path)

                self._plot_count += 1
            except Exception as e:
                print(f"Error generating plot for request date {t_request}: {e}")

    def _reset_metrics(self) -> None:
        """Clear price metrics for the next evaluation period."""
        self._prices_by_request.clear()
        self._price_counts_by_request.clear()

    def _on_step(self) -> bool:
        """Called after each environment step."""
        if not self._initialized:
            self._initialize()

        n_envs = int(self.training_env.num_envs)
        env_idx = self.track_env_idx
        if env_idx < 0 or env_idx >= n_envs:
            return True

        # Get the final requests and prices from the current step
        try:
            final_requests = self._get_attr("last_final_requests", env_idx)
            final_prices = self._get_attr("last_final_prices", env_idx)
        except (AttributeError, IndexError):
            # Attributes might not exist yet
            final_requests = None
            final_prices = None

        # Process each request and its prices
        if final_requests is not None and final_prices is not None and len(final_requests) > 0:
            # We need to figure out which requests correspond to which prices
            # final_prices are ordered by request, with consecutive prices per request's days_stay
            price_idx = 0
            for req in final_requests:
                num_days = req.days_stay

                if price_idx + num_days <= len(final_prices):
                    req_prices = final_prices[price_idx : price_idx + num_days]

                    # Record these prices for this request date and check-in date
                    self.record_prices(req.t_request, req.t_reservation, req_prices)

                    price_idx += num_days

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
