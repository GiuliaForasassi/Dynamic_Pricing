from __future__ import annotations

import csv
from io import StringIO

import mlflow
import numpy as np


class RequestTracker:
    """Conteggi per giorno, mediati sui soli episodi completi della run.

    by_request_day:
        richieste arrivate quel giorno, anche per soggiorni fuori
        dall'intervallo visualizzato.

    by_stay_day:
        richieste che comprendono quella notte, ricevute durante
        l'intero episodio, indipendentemente dall'esito.

    Gli indici interni sono zero-based; CSV e metriche usano giorni 1-based.
    """

    def __init__(self, split: str, log_to_mlflow: bool = True) -> None:
        self.split = split
        self.log_to_mlflow = log_to_mlflow

        self._simulation_days: int | None = None
        self._next_day = 0
        self._episode_id = 0
        self._completed_episodes = 0
        self._pending_requests = 0
        self._pending_episodes = 0

        self._arrivals: np.ndarray | None = None
        self._stays: np.ndarray | None = None
        self._arrivals_sum: np.ndarray | None = None
        self._stays_sum: np.ndarray | None = None

    def record_day(
        self,
        *,
        day_index: int,
        simulation_days: int,
        num_requests: int,
        request_stays: list[tuple[int, int]],
    ) -> None:
        if simulation_days <= 0:
            raise ValueError("Durata della simulazione non positiva.")

        if self._simulation_days is not None:
            if simulation_days != self._simulation_days:
                raise ValueError("Durata cambiata: usare un tracker separato.")

        if day_index != self._next_day or day_index >= simulation_days:
            raise ValueError("Giorno mancante, duplicato o fuori sequenza.")

        if num_requests != len(request_stays):
            raise ValueError("Numero di richieste e soggiorni non coerente.")

        for check_in, nights in request_stays:
            if check_in < day_index or nights <= 0:
                raise ValueError("Soggiorno non valido.")

        if self._simulation_days is None:
            self._simulation_days = simulation_days
            self._arrivals = np.zeros(simulation_days, dtype=np.int64)
            self._stays = np.zeros(simulation_days, dtype=np.int64)
            self._arrivals_sum = np.zeros(simulation_days, dtype=np.int64)
            self._stays_sum = np.zeros(simulation_days, dtype=np.int64)

        assert self._arrivals is not None
        assert self._stays is not None

        self._arrivals[day_index] = num_requests

        for check_in, nights in request_stays:
            stop = min(check_in + nights, simulation_days)
            if check_in < stop:
                self._stays[check_in:stop] += 1

        self._next_day += 1

    def end_episode(self) -> None:
        if self._next_day == 0:
            return

        if self._next_day != self._simulation_days:
            raise ValueError("Impossibile chiudere un episodio incompleto.")

        assert self._arrivals is not None
        assert self._stays is not None
        assert self._arrivals_sum is not None
        assert self._stays_sum is not None

        if self.log_to_mlflow:
            buf = StringIO()
            writer = csv.writer(buf)
            writer.writerow(["day", "requests_arriving", "requests_covering_night"])
            writer.writerows(
                (day + 1, int(arrivals), int(stays))
                for day, (arrivals, stays) in enumerate(
                    zip(self._arrivals, self._stays, strict=True)
                )
            )
            mlflow.log_text(
                buf.getvalue(),
                (f"episodes/{self.split}/episode_{self._episode_id:05d}_request_counts.csv"),
            )

        self._arrivals_sum += self._arrivals
        self._stays_sum += self._stays
        self._completed_episodes += 1

        # Sommiamo gli arrivi, NON le notti richieste.
        self._pending_requests += int(self._arrivals.sum())
        self._pending_episodes += 1

        self._episode_id += 1
        self._clear_episode()

    def log_step_metrics(self, step: int) -> None:
        """Media dei totali degli episodi completati dall'ultimo logging."""
        if self._pending_episodes == 0:
            return

        mean_requests = self._pending_requests / self._pending_episodes

        if self.log_to_mlflow:
            mlflow.log_metric(
                (f"num_requests_per_episode/by_training_timesteps/{self.split}"),
                float(mean_requests),
                step=step,
            )

        # Il checkpoint successivo non deve includere episodi precedenti.
        self._pending_requests = 0
        self._pending_episodes = 0

    def _clear_episode(self) -> None:
        if self._arrivals is not None:
            self._arrivals.fill(0)
        if self._stays is not None:
            self._stays.fill(0)
        self._next_day = 0

    def finalize_training(self) -> None:
        # Non mescoliamo un episodio parziale con quelli completi.
        self._clear_episode()
        # I punti temporali vengono registrati dai callback.
        # Qui non inventiamo un timestep per eventuali dati rimasti.
        self._pending_requests = 0
        self._pending_episodes = 0

        if not self.log_to_mlflow or self._completed_episodes == 0:
            return

        assert self._arrivals_sum is not None
        assert self._stays_sum is not None

        for axis, totals in (
            ("by_request_day", self._arrivals_sum),
            ("by_stay_day", self._stays_sum),
        ):
            for day, total in enumerate(totals):
                mlflow.log_metric(
                    (f"num_requests/{axis}/mean_completed_episodes/{self.split}"),
                    float(total / self._completed_episodes),
                    step=day + 1,
                )
