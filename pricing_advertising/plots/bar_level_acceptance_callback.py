from __future__ import annotations

import tempfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from pricing_advertising.hotel.room import RoomCategory
from pricing_advertising.pricing_model.bar_table import BAR_TABLE


class AcceptanceByBarLevelCallback(BaseCallback):
    """Generates acceptance probability plots by BAR price level for each request date.

    For selected request dates (t_request), generates plots showing the acceptance
    probability for each BAR level (0-28), revealing the price-sensitivity curve
    of the acceptance model learned by the agent.
    """

    def __init__(
        self,
        eval_freq: int = 500,
        n_envs: int = 4,
        log_to_mlflow: bool = True,
        track_env_idx: int = 0,
    ) -> None:
        """Initialize the bar level acceptance callback.

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
        self._initialized = False

        # Track acceptance by BAR level and request date
        # Structure: dict[t_request] -> dict[bar_level] -> {'total': int, 'accepted': int}
        self._bar_level_stats_by_request: dict[date, dict[int, dict[str, int]]] = {}

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

    def _price_to_bar_level(self, price: float, room_category: int) -> int | None:
        """Convert a price to the closest BAR level.

        Args:
            price: The price value.
            room_category: Room category (1, 2, or 3).

        Returns:
            BAR level (0-28+) or None if unable to determine.
        """
        if price < 0:
            return None

        try:
            # Get the BAR table for this room category
            category_name = f"CATEGORY{room_category}"
            category_enum = RoomCategory[category_name]
            prices = BAR_TABLE.get(category_enum, [])

            if not prices:
                return None

            # Find the level with the closest price
            min_diff = float("inf")
            best_level = None
            for level, bar_price in enumerate(prices):
                diff = abs(bar_price - price)
                if diff < min_diff:
                    min_diff = diff
                    best_level = level

            return best_level
        except (KeyError, ValueError):
            return None

    def record_bar_level_request(
        self,
        t_request: date,
        bar_level: int,
        is_accepted: bool,
    ) -> None:
        """Record a request and its acceptance status at a specific BAR level.

        Args:
            t_request: The request date.
            bar_level: The BAR level used for pricing.
            is_accepted: Whether the request was accepted.
        """
        if t_request not in self._bar_level_stats_by_request:
            self._bar_level_stats_by_request[t_request] = {}

        if bar_level not in self._bar_level_stats_by_request[t_request]:
            self._bar_level_stats_by_request[t_request][bar_level] = {"total": 0, "accepted": 0}

        self._bar_level_stats_by_request[t_request][bar_level]["total"] += 1
        if is_accepted:
            self._bar_level_stats_by_request[t_request][bar_level]["accepted"] += 1

    def _generate_plot_for_request_date(self, t_request: date, eval_step: int) -> bytes:
        """Generate an acceptance probability plot by BAR level for a specific request date.

        Args:
            t_request: The request date to generate plot for.
            eval_step: The evaluation step count.

        Returns:
            PNG image as bytes.
        """
        if t_request not in self._bar_level_stats_by_request:
            raise ValueError("Invalid state for plot generation")

        # Collect data for all BAR levels
        bar_levels = []
        acceptance_probabilities = []

        # Get all bar levels that had requests
        levels_with_data = sorted(self._bar_level_stats_by_request[t_request].keys())

        if not levels_with_data:
            raise ValueError(f"No data for request date {t_request}")

        for bar_level in levels_with_data:
            stats = self._bar_level_stats_by_request[t_request][bar_level]
            total = stats["total"]
            accepted = stats["accepted"]

            # Calculate acceptance probability
            prob = accepted / total if total > 0 else 0.0

            bar_levels.append(bar_level)
            acceptance_probabilities.append(prob)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            bar_levels,
            acceptance_probabilities,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=6,
        )
        ax.set_xlabel("BAR Level (Price Level)")
        ax.set_ylabel("Acceptance Probability")
        ax.set_title(
            f"Acceptance Probability by BAR Level - Request Date: {t_request.strftime('%Y-%m-%d')}"
        )
        ax.set_ylim([0, 1])
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
        if not self._initialized:
            return

        eval_step = self.num_timesteps

        for t_request in self.request_dates:
            # Skip if no data for this request date
            if t_request not in self._bar_level_stats_by_request:
                continue

            try:
                # Generate plot
                plot_bytes = self._generate_plot_for_request_date(t_request, eval_step)

                # Create artifact path with request date as folder and step count in filename
                request_date_str = t_request.strftime("%Y-%m-%d")
                filename = f"{eval_step:08d}_eval_timesteps.png"
                artifact_path = f"bar_level_acceptance_plots/{request_date_str}"

                if self.log_to_mlflow:
                    # Create temporary directory and file with the desired filename
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_file_path = Path(tmpdir) / filename
                        tmp_file_path.write_bytes(plot_bytes)
                        mlflow.log_artifact(str(tmp_file_path), artifact_path=artifact_path)

                self._plot_count += 1
            except Exception as e:
                print(
                    f"Error generating bar level acceptance plot for request date {t_request}: {e}"
                )

    def _reset_metrics(self) -> None:
        """Clear bar level acceptance metrics for the next evaluation period."""
        self._bar_level_stats_by_request.clear()

    def _on_step(self) -> bool:
        """Called after each environment step."""
        if not self._initialized:
            self._initialize()

        n_envs = int(self.training_env.num_envs)
        env_idx = self.track_env_idx
        if env_idx < 0 or env_idx >= n_envs:
            return True

        # Get valid requests (all requests with prices offered) and final (accepted) requests
        try:
            valid_requests = self._get_attr("last_valid_requests", env_idx)
            valid_prices = self._get_attr("last_valid_prices", env_idx)
            final_requests = self._get_attr("last_final_requests", env_idx)
        except (AttributeError, IndexError):
            valid_requests = None
            valid_prices = None
            final_requests = None

        # Create a set of accepted request identifiers for easy lookup
        accepted_requests_set = set()
        if final_requests is not None and len(final_requests) > 0:
            for req in final_requests:
                # Use (t_request, t_reservation, category) as identifier
                accepted_requests_set.add((req.t_request, req.t_reservation, req.category))

        # Process all valid requests
        if valid_requests is not None and valid_prices is not None and len(valid_requests) > 0:
            for req, total_price in zip(valid_requests, valid_prices, strict=True):
                # Convert price to BAR level
                bar_level = self._price_to_bar_level(total_price, req.category.value)

                if bar_level is None:
                    continue

                # Check if this request was accepted
                is_accepted = (
                    req.t_request,
                    req.t_reservation,
                    req.category,
                ) in accepted_requests_set

                # Record the request
                self.record_bar_level_request(req.t_request, bar_level, is_accepted)

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
