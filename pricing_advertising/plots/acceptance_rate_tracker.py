from __future__ import annotations

from dataclasses import dataclass

import mlflow


# -------- ACCEPTANCE COUNTS --------
@dataclass(frozen=True)
class AcceptanceCounts:
    requests: int = 0  # Total number of requests received
    offers: int = 0  # Total number of offers made
    price_accepted: int = 0  # Total number of offers accepted based on price
    confirmed: int = 0  # Total number of confirmed requests

    # -------- Validation --------
    # requests >= offers >= price_accepted >= confirmed >= 0
    def __post_init__(self) -> None:
        if not (0 <= self.confirmed <= self.price_accepted <= self.offers <= self.requests):
            raise ValueError("Acceptance counts are inconsistent.")

    # -------- Arithmetic Operations --------
    def __add__(self, other: AcceptanceCounts) -> AcceptanceCounts:
        return AcceptanceCounts(
            requests=self.requests + other.requests,
            offers=self.offers + other.offers,
            price_accepted=self.price_accepted + other.price_accepted,
            confirmed=self.confirmed + other.confirmed,
        )

    # -------- Rate Computation --------
    def rates(self) -> dict[str, float]:
        return {
            # Confirmation rate: confirmed requests / total requests
            # How many of the total requests were confirmed and began effective bookings?
            "confirmation_rate": (
                self.confirmed / self.requests if self.requests > 0 else float("nan")
            ),
            # Price acceptance rate: price_accepted / offers
            # Between clients who received an offer, how many accepted?
            "price_acceptance_rate": (
                self.price_accepted / self.offers if self.offers > 0 else float("nan")
            ),
        }


# -------- ACCEPTANCE RATE TRACKER --------
# Tracker for acceptance rates based on daily request and offer counts.
class AcceptanceRateTracker:
    """Tassi per giorno di richiesta e per timestep di training.

    I riepiloghi includono soltanto episodi completi.

    - by_training_timesteps:
      rapporto dei conteggi degli episodi completati dall'ultimo log.
      Training: un episodio. Evaluation: un intero round.

    - by_request_day_pooled_completed_episodes:
      rapporto dei conteggi per ciascun giorno, aggregati su tutti gli
      episodi completi registrati durante la run, anche di checkpoint diversi.
    """

    def __init__(self, split: str, log_to_mlflow: bool = True) -> None:
        self.split = split
        self.log_to_mlflow = log_to_mlflow

        self._episode: list[AcceptanceCounts] = []
        self._expected_days: int | None = None

        self._by_day: dict[int, AcceptanceCounts] = {}
        self._pending_counts: AcceptanceCounts | None = None

    # --------- RECORD DAY --------
    def record_day(
        self,
        *,
        day_index: int,
        simulation_days: int,
        num_requests: int,
        num_offers: int,
        num_price_accepted: int,
        num_confirmed: int,
    ) -> None:
        # - Validate input and sequence of days before recording.
        if simulation_days <= 0:
            raise ValueError("Simulation days must be positive.")

        if self._expected_days is None:
            self._expected_days = simulation_days
        elif simulation_days != self._expected_days:
            raise ValueError("Simulation duration changed during the episode.")

        # Each day must be recorded, even with all counts set to zero.
        if day_index != len(self._episode) or day_index >= simulation_days:
            raise ValueError("Missing, duplicate, or out-of-sequence day.")

        # - Record the day's counts.
        self._episode.append(
            AcceptanceCounts(
                requests=num_requests,
                offers=num_offers,
                price_accepted=num_price_accepted,
                confirmed=num_confirmed,
            )
        )

    # --------- END EPISODE --------
    def end_episode(self) -> AcceptanceCounts | None:
        if not self._episode:
            return None

        if len(self._episode) != self._expected_days:
            raise ValueError("Cannot close an incomplete episode.")

        totals = AcceptanceCounts()

        # - Aggregate counts for the episode and update pending counts.
        for day_index, counts in enumerate(self._episode):
            totals = totals + counts
            self._by_day[day_index] = self._by_day.get(day_index, AcceptanceCounts()) + counts

        if self._pending_counts is None:
            self._pending_counts = totals
        else:
            self._pending_counts = self._pending_counts + totals

        self._episode = []
        self._expected_days = None
        return totals

    # --------- LOG RATES --------
    def _log_rates(
        self,
        counts: AcceptanceCounts,
        *,
        axis: str,
        step: int,
    ) -> None:
        if not self.log_to_mlflow:
            return

        # - Log the acceptance rates for the given counts.
        for name, value in counts.rates().items():
            mlflow.log_metric(
                f"{name}/{axis}/{self.split}",
                value,
                step=step,
            )

    # --------- LOG STEP METRICS --------
    # Log and clear the counts of completed episodes pending.
    def log_step_metrics(self, step: int) -> None:
        if self._pending_counts is None:
            return

        self._log_rates(
            self._pending_counts,
            axis="by_training_timesteps",
            step=step,
        )
        self._pending_counts = None

    # --------- FINALIZE TRAINING --------
    def finalize_training(self) -> None:
        # An episode interrupted by the end of training does not enter
        # the summaries of completed episodes.

        # Clear any ongoing episode as it will not be included in the summaries.
        self._episode = []
        self._expected_days = None

        # Log the acceptance rates for each day of completed episodes.
        for day_index, counts in sorted(self._by_day.items()):
            self._log_rates(
                counts,
                axis="by_request_day_pooled_completed_episodes",
                step=day_index + 1,
            )
