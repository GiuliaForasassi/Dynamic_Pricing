from __future__ import annotations

import csv
from datetime import date
from io import StringIO
from typing import TYPE_CHECKING

import mlflow
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pricing_advertising.requests.request import Request


class PriceTracker:
    """Prezzo delle prenotazioni confermate per camera-notte.

    Include soltanto le notti nell'orizzonte principale.
    Aggrega importi e conteggi, non medie di medie.
    Usa esclusivamente episodi completi.
    """

    def __init__(self, split: str, log_to_mlflow: bool = True) -> None:
        self.split = split
        self.log_to_mlflow = log_to_mlflow

        self._start: date | None = None
        self._days: int | None = None
        self._next_day = 0
        self._episode_id = 0
        self._completed_episodes = 0

        self._revenue: np.ndarray | None = None
        self._nights: np.ndarray | None = None
        self._revenue_sum: np.ndarray | None = None
        self._nights_sum: np.ndarray | None = None

        self._pending_revenue = 0.0
        self._pending_nights = 0
        self._pending_episodes = 0

    @staticmethod
    def _ratio(revenue: float, nights: int) -> float:
        return float(revenue / nights) if nights > 0 else float("nan")

    def record_day(
        self,
        *,
        day_index: int,
        start_simulation: date,
        simulation_days: int,
        confirmed_requests: Sequence[Request],
        confirmed_nightly_prices: np.ndarray,
    ) -> None:
        if simulation_days <= 0:
            raise ValueError("Durata della simulazione non positiva.")

        if self._days is not None and (
            simulation_days != self._days or start_simulation != self._start
        ):
            raise ValueError("Calendario cambiato: usare un tracker separato.")

        if day_index != self._next_day or not 0 <= day_index < simulation_days:
            raise ValueError("Giorno mancante, duplicato o fuori sequenza.")

        requests = list(confirmed_requests)
        lengths = [int(request.days_stay) for request in requests]

        if any(
            nights <= 0 or nights != request.days_stay
            for request, nights in zip(requests, lengths, strict=True)
        ):
            raise ValueError("Durata del soggiorno non valida.")

        prices = np.asarray(confirmed_nightly_prices, dtype=float)
        if prices.ndim != 1 or prices.size != sum(lengths):
            raise ValueError("Prezzi notturni non allineati alle richieste.")

        if not np.all(np.isfinite(prices)) or np.any(prices < 0):
            raise ValueError("Prezzi non validi: devono essere finiti e non negativi.")

        # Costruiamo prima i contributi del giorno.
        # Nessun dato persistente viene modificato prima delle verifiche.
        daily_revenue = np.zeros(simulation_days, dtype=float)
        daily_nights = np.zeros(simulation_days, dtype=np.int64)

        price_index = 0
        for request, nights in zip(requests, lengths, strict=True):
            request_prices = prices[price_index : price_index + nights]
            price_index += nights

            offset = (request.t_reservation - start_simulation).days
            first = max(0, -offset)
            stop = min(nights, simulation_days - offset)

            if first < stop:
                target = slice(offset + first, offset + stop)
                daily_revenue[target] += request_prices[first:stop]
                daily_nights[target] += 1

        if self._days is None:
            self._start = start_simulation
            self._days = simulation_days
            self._revenue = np.zeros(simulation_days, dtype=float)
            self._nights = np.zeros(simulation_days, dtype=np.int64)
            self._revenue_sum = np.zeros(simulation_days, dtype=float)
            self._nights_sum = np.zeros(simulation_days, dtype=np.int64)

        assert self._revenue is not None
        assert self._nights is not None

        self._revenue += daily_revenue
        self._nights += daily_nights
        self._next_day += 1

    def end_episode(self) -> None:
        if self._next_day == 0:
            return

        if self._next_day != self._days:
            raise ValueError("Impossibile chiudere un episodio incompleto.")

        assert self._revenue is not None
        assert self._nights is not None
        assert self._revenue_sum is not None
        assert self._nights_sum is not None

        if self.log_to_mlflow:
            buf = StringIO()
            writer = csv.writer(buf)
            writer.writerow(["day", "booked_revenue", "room_nights", "booked_mean_price"])
            for day, (revenue, nights) in enumerate(
                zip(self._revenue, self._nights, strict=True),
                start=1,
            ):
                writer.writerow(
                    [
                        day,
                        float(revenue),
                        int(nights),
                        self._ratio(float(revenue), int(nights)),
                    ]
                )
            mlflow.log_text(
                buf.getvalue(),
                f"episodes/{self.split}/episode_{self._episode_id:05d}_price.csv",
            )

        self._revenue_sum += self._revenue
        self._nights_sum += self._nights
        self._completed_episodes += 1

        self._pending_revenue += float(self._revenue.sum())
        self._pending_nights += int(self._nights.sum())
        self._pending_episodes += 1

        self._episode_id += 1
        self._clear_episode()

    def log_step_metrics(self, step: int) -> None:
        if self._pending_episodes == 0:
            return

        if self.log_to_mlflow:
            mlflow.log_metric(
                f"booked_mean_price/by_training_timesteps/{self.split}",
                self._ratio(self._pending_revenue, self._pending_nights),
                step=step,
            )

        self._pending_revenue = 0.0
        self._pending_nights = 0
        self._pending_episodes = 0

    def _clear_episode(self) -> None:
        if self._revenue is not None:
            self._revenue.fill(0)
        if self._nights is not None:
            self._nights.fill(0)
        self._next_day = 0

    def finalize_training(self) -> None:
        self._clear_episode()
        self._pending_revenue = 0.0
        self._pending_nights = 0
        self._pending_episodes = 0

        if not self.log_to_mlflow or self._completed_episodes == 0:
            return

        assert self._revenue_sum is not None
        assert self._nights_sum is not None

        for day, (revenue, nights) in enumerate(
            zip(self._revenue_sum, self._nights_sum, strict=True),
            start=1,
        ):
            mlflow.log_metric(
                (f"booked_mean_price/by_stay_day/pooled_completed_episodes/{self.split}"),
                self._ratio(float(revenue), int(nights)),
                step=day,
            )
