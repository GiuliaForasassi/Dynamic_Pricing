from typing import Any

from pricing_advertising.plots.acceptance_rate_tracker import AcceptanceRateTracker
from pricing_advertising.plots.occupancy_tracker import OccupancyTracker
from pricing_advertising.plots.price_tracker import PriceTracker
from pricing_advertising.plots.request_tracker import RequestTracker
from pricing_advertising.plots.reward_tracker import RewardTracker


# - Used for random policy evaluation -
class EpisodeMetrics:
    """Feed the same episode metrics to trackers regardless of the evaluated policy."""

    def __init__(self, split: str, log_to_mlflow: bool = True) -> None:
        self.occupancy = OccupancyTracker(split=split, log_to_mlflow=log_to_mlflow)
        self.requests = RequestTracker(split=split, log_to_mlflow=log_to_mlflow)
        self.reward = RewardTracker(split=split, log_to_mlflow=log_to_mlflow)
        self.price = PriceTracker(split=split, log_to_mlflow=log_to_mlflow)
        self.acceptance = AcceptanceRateTracker(split=split, log_to_mlflow=log_to_mlflow)
        self._trackers = (
            self.occupancy,
            self.requests,
            self.reward,
            self.price,
            self.acceptance,
        )

    def record_day(self, metrics: dict[str, Any]) -> None:
        self.occupancy.record_day(
            day_index=int(metrics["day_index"]),
            simulation_days=int(metrics["simulation_days"]),
            occupancy_pct=float(metrics["occupancy_pct"]),
        )
        self.requests.record_day(
            day_index=int(metrics["day_index"]),
            simulation_days=int(metrics["simulation_days"]),
            num_requests=int(metrics["num_requests"]),
            request_stays=metrics["request_stays"],
        )
        self.reward.record_day(
            day_index=int(metrics["day_index"]),
            simulation_days=int(metrics["simulation_days"]),
            reward=float(metrics["reward"]),
        )
        self.price.record_day(
            day_index=int(metrics["day_index"]),
            start_simulation=metrics["start_simulation"],
            simulation_days=int(metrics["simulation_days"]),
            confirmed_requests=metrics["confirmed_requests"],
            confirmed_nightly_prices=metrics["confirmed_nightly_prices"],
        )
        self.acceptance.record_day(
            day_index=int(metrics["day_index"]),
            simulation_days=int(metrics["simulation_days"]),
            num_requests=int(metrics["num_requests"]),
            num_offers=int(metrics["num_offers"]),
            num_price_accepted=int(metrics["num_price_accepted"]),
            num_confirmed=int(metrics["num_confirmed"]),
        )

    def end_episode(self) -> None:
        for tracker in self._trackers:
            tracker.end_episode()

    def log_step_metrics(self, step: int) -> None:
        for tracker in self._trackers:
            tracker.log_step_metrics(step=step)

    def finalize(self) -> None:
        for tracker in self._trackers:
            tracker.finalize_training()
