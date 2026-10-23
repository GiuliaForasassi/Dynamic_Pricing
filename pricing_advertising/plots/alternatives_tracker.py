from __future__ import annotations

from typing import Any

import mlflow
import numpy as np

from pricing_advertising.plots.alternatives_plots import (
    ALTERNATIVE_ROWS,
    CONFIRMED,
    N_COLUMNS,
    N_REQUEST_COUNTERS,
    ORIGINAL_ROW,
    OUTSIDE_WINDOW,
    PRICE_ACCEPTED,
    PROPOSED,
    RECOVERY,
    REFERENCE_REVENUE,
    REQUESTS,
    REVENUE,
    UNAVAILABLE,
    log_alternative_figures,
)
from pricing_advertising.requests.request import (
    ALTERNATIVE_TYPES,
    CATEGORY_ALTERNATIVES,
    OFFER_TYPES,
    PERIOD_ALTERNATIVES,
)

_ROW_BY_TYPE = {name: row for row, name in enumerate(OFFER_TYPES)}
_ROWS_BY_GROUP = {
    "original": [ORIGINAL_ROW],
    "alternative": ALTERNATIVE_ROWS,
    "category": [_ROW_BY_TYPE[name] for name in CATEGORY_ALTERNATIVES],
    "period": [_ROW_BY_TYPE[name] for name in PERIOD_ALTERNATIVES],
}
# Columns of the per-check-in-day array
_N_DAY_COLUMNS = 3


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else float("nan")


# -------- ALTERNATIVES TRACKER --------
# Tracks how often alternatives (different room category or different period) are proposed,
# which kinds are proposed and how many are accepted and confirmed.
#
# Offer types come from HotelOffer.offer_type: "original" (the request as is) and 7 alternatives.
# An alternative can be proposed at two stages:
#   - "initial": the requested room/period is not available, so the offer is built with an alternative;
#   - "recovery": the price of the offer was accepted but the booking failed (another guest took the last room).
#
# Only completed episodes enter the summaries.
#   - log_step_metrics: per-episode means (counts) and pooled rates since the last log.
#   - finalize_training: figures with means over all the completed episodes. The by-day figure is
#     indexed by the check-in day of the original request (to compare it with the occupancy of
#     that day); offers whose check-in is after the simulation end only enter the totals.
class AlternativesTracker:
    def __init__(self, split: str, log_to_mlflow: bool = True) -> None:
        self.split = split
        self.log_to_mlflow = log_to_mlflow

        self._simulation_days: int | None = None
        self._next_day = 0

        # Current episode
        self._totals = np.zeros((len(OFFER_TYPES), N_COLUMNS))
        self._requests = np.zeros(N_REQUEST_COUNTERS)
        self._by_checkin: np.ndarray | None = None
        self._occupancy: np.ndarray | None = None

        # Sum over all the completed episodes
        self._completed_episodes = 0
        self._totals_sum = np.zeros_like(self._totals)
        self._requests_sum = np.zeros_like(self._requests)
        self._by_checkin_sum: np.ndarray | None = None
        self._occupancy_sum: np.ndarray | None = None

        # Sum over the episodes completed since the last log
        self._pending_episodes = 0
        self._pending_totals = np.zeros_like(self._totals)
        self._pending_requests = np.zeros_like(self._requests)

    # --------- RECORD DAY --------
    def record_day(
        self,
        *,
        day_index: int,
        simulation_days: int,
        num_requests: int,
        occupancy_pct: float,
        offer_stats: dict[str, Any],
    ) -> None:
        if simulation_days <= 0:
            raise ValueError("Simulation days must be positive.")

        if self._simulation_days is not None and simulation_days != self._simulation_days:
            raise ValueError("Simulation duration changed: use a separate tracker.")

        if day_index != self._next_day or day_index >= simulation_days:
            raise ValueError("Missing, duplicate, or out-of-sequence day.")

        events = offer_stats["events"]
        outside_window = int(offer_stats["num_outside_booking_window"])
        unavailable = int(offer_stats["num_unavailable"])
        initial_offers = sum(1 for event in events if event["stage"] == "initial")
        if num_requests != outside_window + unavailable + initial_offers:
            raise ValueError("Offer statistics inconsistent with the number of requests.")

        if self._simulation_days is None:
            self._simulation_days = simulation_days
        if self._by_checkin is None:
            self._by_checkin = np.zeros((simulation_days, len(OFFER_TYPES), _N_DAY_COLUMNS))
            self._occupancy = np.zeros(simulation_days)
        assert self._by_checkin is not None
        assert self._occupancy is not None

        for event in events:
            row = _ROW_BY_TYPE[event["type"]]
            accepted = bool(event["price_accepted"])
            confirmed = bool(event["confirmed"])
            if confirmed and not accepted:
                raise ValueError("A confirmed offer must have an accepted price.")

            self._totals[row, PROPOSED] += 1
            self._totals[row, PRICE_ACCEPTED] += accepted
            self._totals[row, CONFIRMED] += confirmed
            self._totals[row, RECOVERY] += event["stage"] == "recovery"
            self._totals[row, REVENUE] += float(event["price"])
            self._totals[row, REFERENCE_REVENUE] += float(event["reference_price"])

            check_in = int(event["check_in_index"])
            if 0 <= check_in < simulation_days:
                self._by_checkin[check_in, row, PROPOSED] += 1
                self._by_checkin[check_in, row, PRICE_ACCEPTED] += accepted
                self._by_checkin[check_in, row, CONFIRMED] += confirmed

        self._requests[REQUESTS] += num_requests
        self._requests[OUTSIDE_WINDOW] += outside_window
        self._requests[UNAVAILABLE] += unavailable
        self._occupancy[day_index] = float(occupancy_pct)
        self._next_day += 1

    # --------- END EPISODE --------
    def end_episode(self) -> None:
        if self._next_day == 0:
            return

        if self._next_day != self._simulation_days:
            raise ValueError("Cannot close an incomplete episode.")

        assert self._by_checkin is not None
        assert self._occupancy is not None

        if self._by_checkin_sum is None or self._occupancy_sum is None:
            self._by_checkin_sum = np.zeros_like(self._by_checkin)
            self._occupancy_sum = np.zeros_like(self._occupancy)

        self._totals_sum += self._totals
        self._requests_sum += self._requests
        self._by_checkin_sum += self._by_checkin
        self._occupancy_sum += self._occupancy
        self._completed_episodes += 1

        self._pending_totals += self._totals
        self._pending_requests += self._requests
        self._pending_episodes += 1

        self._clear_episode()

    def _clear_episode(self) -> None:
        self._totals.fill(0)
        self._requests.fill(0)
        if self._by_checkin is not None and self._occupancy is not None:
            self._by_checkin.fill(0)
            self._occupancy.fill(0)
        self._next_day = 0

    # --------- LOG STEP METRICS --------
    # Log and clear the episodes completed since the last log.
    # Rates with no denominator (e.g. the alternatives of a run that never proposes them) are not logged.
    def log_step_metrics(self, step: int) -> None:
        if self._pending_episodes == 0:
            return

        episodes = self._pending_episodes
        totals = self._pending_totals
        requests = self._pending_requests

        if self.log_to_mlflow:
            split = self.split
            suffix = f"by_training_timesteps/{split}"
            metrics: dict[str, float] = {}

            # How often alternatives are proposed
            for group in ("alternative", "category", "period"):
                proposed = float(totals[_ROWS_BY_GROUP[group], PROPOSED].sum())
                metrics[f"alternatives_proposed/{group}/{suffix}"] = proposed / episodes
            metrics[f"alternatives_proposed/after_failed_booking/{suffix}"] = (
                float(totals[ALTERNATIVE_ROWS, RECOVERY].sum()) / episodes
            )
            metrics[f"alternative_proposal_rate/{suffix}"] = _ratio(
                float(totals[ALTERNATIVE_ROWS, PROPOSED].sum()), float(requests[REQUESTS])
            )

            # Which kinds of alternatives are proposed
            for name in ALTERNATIVE_TYPES:
                metrics[f"alternatives_proposed_by_type/{name}/{suffix}"] = (
                    float(totals[_ROW_BY_TYPE[name], PROPOSED]) / episodes
                )

            # How many offers are accepted and confirmed
            for group, rows in _ROWS_BY_GROUP.items():
                proposed = float(totals[rows, PROPOSED].sum())
                metrics[f"offer_price_acceptance_rate/{group}/{suffix}"] = _ratio(
                    float(totals[rows, PRICE_ACCEPTED].sum()), proposed
                )
                metrics[f"offer_confirmation_rate/{group}/{suffix}"] = _ratio(
                    float(totals[rows, CONFIRMED].sum()), proposed
                )

            # How much the alternatives are worth
            total_revenue = float(totals[:, REVENUE].sum())
            alternative_revenue = float(totals[ALTERNATIVE_ROWS, REVENUE].sum())
            metrics[f"alternative_revenue/{suffix}"] = alternative_revenue / episodes
            metrics[f"alternative_revenue_share/{suffix}"] = _ratio(
                alternative_revenue, total_revenue
            )
            metrics[f"alternative_price_ratio/{suffix}"] = _ratio(
                alternative_revenue, float(totals[ALTERNATIVE_ROWS, REFERENCE_REVENUE].sum())
            )

            # Requests that end without any offer
            metrics[f"requests_without_offer/no_availability/{suffix}"] = (
                float(requests[UNAVAILABLE]) / episodes
            )
            metrics[f"requests_without_offer/outside_window/{suffix}"] = (
                float(requests[OUTSIDE_WINDOW]) / episodes
            )

            for key, value in metrics.items():
                if not np.isnan(value):
                    mlflow.log_metric(key, value, step=step)

        self._pending_totals.fill(0)
        self._pending_requests.fill(0)
        self._pending_episodes = 0

    # --------- FINALIZE TRAINING --------
    def finalize_training(self) -> None:
        # An episode interrupted by the end of training does not enter the summaries.
        self._clear_episode()
        self._pending_totals.fill(0)
        self._pending_requests.fill(0)
        self._pending_episodes = 0

        if not self.log_to_mlflow or self._completed_episodes == 0:
            return

        assert self._by_checkin_sum is not None
        assert self._occupancy_sum is not None

        episodes = self._completed_episodes
        log_alternative_figures(
            self.split,
            totals=self._totals_sum / episodes,
            by_checkin=self._by_checkin_sum / episodes,
            occupancy_pct=self._occupancy_sum / episodes,
            requests=self._requests_sum / episodes,
            num_episodes=episodes,
        )
