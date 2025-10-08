import random
from datetime import date, timedelta

import numpy as np
from loguru import logger
from numpy.random import Generator, default_rng
from scipy.stats import norm, rv_continuous

from pricing_advertising.exceptions.exceptions import InvalidReservationWindowError
from pricing_advertising.requests.request import Request
from pricing_advertising.utils.constants import (
    MAX_DAYS_STAY,
    MAX_GUESTS,
    MIN_GUESTS,
    MIN_NUM_REQUESTS_FACTOR,
    SEEDS,
    STD_FACTOR,
)


class RequestGenerator:
    def __init__(
        self,
        avg_num_requests: int,
        start_simulation: date,
        end_simulation: date,
        seasonality_func: np.ndarray | None = None,
        distribution: rv_continuous | None = None,
        seed: int | None = None,
    ) -> None:
        if not isinstance(end_simulation, date):
            raise ValueError("The end_simulation must be a date object")

        if avg_num_requests < 0:
            raise ValueError("The average number of requests must be greater than 0")

        self.seed = seed
        self.rng: Generator = default_rng(seed)
        self.py_rng = random.Random(seed)
        self.avg_num_requests = int(avg_num_requests)
        self.start_simulation = start_simulation
        self.end_simulation = end_simulation

        self.seasonality_func = (
            seasonality_func.astype(float)
            if seasonality_func is not None
            else np.ones(365, dtype=float)
        ).astype(float)
        if self.seasonality_func.shape != (365,):
            raise ValueError("`seasonality_func` must be a vector of length 365")

        # If no distribution is provided, we use a normal distribution (loc: mean, scale: std)
        self.base_dist = (
            distribution
            if distribution is not None
            else norm(
                loc=self.avg_num_requests,
                scale=max(1e-6, STD_FACTOR * max(1, self.avg_num_requests)),
            )
        )

    # ============ METHODS TO USE IN THE ENV ============
    def generate_requests(self, today: date) -> np.ndarray:
        """Genera SOLO le richieste del giorno `today` (array dtype=object, anche vuoto)."""
        return np.array(self.generate_requests_arriving_today(today), dtype=object)

    def generate_requests_arriving_today(self, today: date) -> list[Request]:
        if not isinstance(today, date):
            raise ValueError("today must be a date")
        days_left = (self.end_simulation - today).days
        if days_left <= 1:
            # No requests can be made on the last day or if there's no time left
            return []

        # Sample the number of requests for today
        num_requests = self._sample_num_requests(today)
        requests: list[Request] = []

        # generation of requests
        for _ in range(num_requests):
            request = self._generate_request(today)
            if request is not None:
                requests.append(request)
        return requests

    # ============ SAMPLING HELPERS ============
    def _sample_num_requests(self, today: date, size: int = 1) -> int:
        day_of_year = today.timetuple().tm_yday - 1  # 0-indexed: 0-364
        # Get the seasonality factor for the current day
        seasonality_factor = float(self.seasonality_func[day_of_year])
        # Calculate the number of requests based on the average number of requests and the seasonality factor
        adjusted_mean = max(0.0, self.avg_num_requests * seasonality_factor)
        # Create a normal distribution with the adjusted mean and standard deviation
        distribution = norm(
            loc=adjusted_mean, scale=max(1e-6, STD_FACTOR * max(1.0, adjusted_mean))
        )
        # Sample from the distribution
        sample = distribution.rvs(size=1, random_state=self.rng)[0]
        # Min number of requests
        min_num_requests = int(max(0, MIN_NUM_REQUESTS_FACTOR * self.avg_num_requests))
        # Ensure the number of requests is an integer and non-negative
        num_requests = int(np.round(max(sample, min_num_requests)))
        return num_requests

    def _generate_request(self, today: date) -> Request | None:
        maximum_advance = (self.end_simulation - self.start_simulation).days
        # Possible advances are between 1 and maximum_advance - 1
        possible_advances = np.arange(1, maximum_advance, dtype=int)
        # Weights for the advances are linearly decreasing
        weights = np.arange(len(possible_advances), 0, -1, dtype=float)
        # Normalize the weights to get probabilities
        advances_probs = weights / weights.sum()
        # Sample a random advance from the possible advances with the given probabilities
        advance_days = int(self.rng.choice(possible_advances, p=advances_probs))
        # Reservation date
        t_reservation = today + timedelta(days=advance_days)
        # Days that can be booked
        remaining_days_from_reservation = (self.end_simulation - t_reservation).days
        max_stay = int(min(MAX_DAYS_STAY, max(1, remaining_days_from_reservation)))
        # Ensure there's enough space for at least one night
        if max_stay < 1:
            return None
        days_stay = int(self.rng.integers(1, max_stay + 1))  # [1, max_stay]
        # Number of guests can be 1, 2 or 3
        num_guests = self.rng.integers(MIN_GUESTS, MAX_GUESTS)
        return Request(
            t_request=today,
            t_reservation=t_reservation,
            days_stay=days_stay,
            num_guests=num_guests,
        )


# ============ TEST ============
if __name__ == "__main__":
    # ---- Parametri di test ----
    END_SIM = date(2025, 1, 31)
    START_SIM = date(2025, 1, 1)
    SEED = 42
    AVG_REQ = 12

    # Stagionalità fittizia (tutti 1.0)
    seasonality = np.ones(365, dtype=float)

    rg = RequestGenerator(
        avg_num_requests=AVG_REQ,
        start_simulation=START_SIM,
        end_simulation=END_SIM,
        seasonality_func=seasonality,
        seed=SEED,
    )

    def check_requests(today: date) -> np.ndarray:
        """Genera e valida le richieste per 'today'."""
        arr = rg.generate_requests(today)
        logger.info(f"[{today}] generated {len(arr)} requests")

        # Deve essere un ndarray (anche vuoto), dtype=object
        assert isinstance(arr, np.ndarray), "generate_requests deve restituire un np.ndarray"
        assert arr.dtype == object, f"dtype atteso object, trovato {arr.dtype}"

        for r in arr:
            # Tipi base
            assert hasattr(r, "t_request"), "Request mal formata: manca t_request"
            assert hasattr(r, "t_reservation"), "Request mal formata: manca t_reservation"
            assert r.t_request == today, "t_request deve essere il 'today' passato al generatore"

            # Vincoli temporali
            assert today < r.t_reservation <= END_SIM, (
                "t_reservation deve essere dopo oggi e entro end_simulation"
            )

            # days_stay: 'giorni da considerare dopo la data iniziale'
            # -> consentiamo sia 0 che >=1, ma mai negativo
            assert r.days_stay >= 0, "days_stay non può essere negativo"

            # Non deve sforare l’orizzonte (inclusivo sul calendario)
            remaining_after_res = (END_SIM - r.t_reservation).days
            assert r.days_stay <= remaining_after_res, (
                f"days_stay={r.days_stay} sfora end_sim: restano {remaining_after_res} giorni dopo la reservation"
            )

            # Rispetta i limiti globali

            assert r.days_stay <= MAX_DAYS_STAY, "days_stay oltre MAX_DAYS_STAY"
            assert MIN_GUESTS <= r.num_guests <= MAX_GUESTS, "num_guests fuori range"

        return arr

    # ---- 1) Giorno “normale” nel mezzo dell’orizzonte ----
    _ = check_requests(date(2025, 1, 10))

    # ---- 2) Penultimo giorno utile (edge case) ----
    _ = check_requests(
        END_SIM - timedelta(days=2)
    )  # 29 gen → ci deve essere ancora qualche richiesta possibile

    # ---- 3) Ultimo giorno (nessuna richiesta) ----
    arr_last = rg.generate_requests(END_SIM)
    logger.info(f"[{END_SIM}] generated {len(arr_last)} requests (atteso 0)")
    assert isinstance(arr_last, np.ndarray)
    assert arr_last.dtype == object
    assert len(arr_last) == 0, "All'ultimo giorno non dovrebbero esserci richieste generabili"

    # ---- 4) Piccolo smoke test su più giorni consecutivi ----
    total = 0
    today = date(2025, 1, 1)
    while today < END_SIM:
        total += len(rg.generate_requests(today))
        today += timedelta(days=1)
    logger.info(f"Total requests over the month: {total}")

    logger.info("RequestGenerator quick test: OK ✅")
