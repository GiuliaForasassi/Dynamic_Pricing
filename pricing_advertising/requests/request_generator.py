import random
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

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
        extended_end_simulation: date,
        seasonality_func: np.ndarray | None = None,
        distribution: rv_continuous | None = None,
        seed: int | None = None,
        # constructuctor parameter: if True, precompute all requests at the start
        # If true, all requests for the entire simulation period are precomputed during initialization.
        # If false (on-demand generation), requests are generated as needed for each day (old logic)
        precompute: bool = True,
    ) -> None:
        if not isinstance(end_simulation, date) or not isinstance(start_simulation, date):
            raise ValueError("The start_simulation and the end_simulation must be date objects")

        if avg_num_requests < 0:
            raise ValueError("The average number of requests must be greater than 0")

        # Check using differences in days (no operators < is on dates -> Pylance)
        delta_main = (end_simulation - start_simulation).days
        delta_extended = (extended_end_simulation - start_simulation).days
        if delta_main <= 0 or delta_extended < 0:
            raise ValueError(
                "Must be valid: start_simulation < end_simulation < extendend_end_simulation"
            )

        self.seed = seed
        self.rng: Generator = default_rng(seed)
        self.py_rng = random.Random(seed)
        self.avg_num_requests = int(avg_num_requests)
        self.start_simulation = start_simulation
        self.end_simulation = end_simulation
        self.extended_end_simulation = extended_end_simulation

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

        # ============ PRECOMPUTATION (if needed) ============
        # Dictionary to store precomputed requests by day
        self.requests_by_day: dict[date, list[Request]] = {}
        if precompute:
            self._pre_generate_all_requests()

    # ============ METHODS TO USE IN THE ENV ============
    def fetch_requests_for_date(self, today: date) -> np.ndarray:
        return np.array(self.requests_by_day.get(today, []), dtype=object)

    def ensure_precomputed(self) -> None:
        """Ensure that all requests have been precomputed."""
        if not getattr(self, "requests_by_day", {}):
            self._pre_generate_all_requests()

    def regenerate(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
            # re-seed
            self.rng = default_rng(self.seed)
            self.py_rng = random.Random(self.seed)
            # Clear old requests and generate again
            self.requests_by_day.clear()
            self._pre_generate_all_requests()

    # ============ PRECOMPUTATION HELPERS ============
    def _pre_generate_all_requests(self) -> None:
        current = self.start_simulation
        while current < self.end_simulation:  # t_request must be < end_simulation
            num_requests = self._sample_num_requests(current)
            requests: list[Request] = []
            for _ in range(num_requests):
                request = self._generate_request(current)
                if request is not None:
                    requests.append(request)
            # Store only if there are requests
            if requests:
                self.requests_by_day[current] = requests
            current += timedelta(days=1)

    # ============ SAMPLING THE NUMBER OF REQUESTS ============
    def _sample_num_requests(self, today: date) -> int:
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

    # ============ ONE REQUEST GENERATION ============
    def _generate_request(self, today: date) -> Request | None:
        remaining_from_today = (self.extended_end_simulation - today).days
        if remaining_from_today <= 1:
            return None  # No more requests can be made

        # Possible advances are between 1 and maximum_advance - 1
        possible_advances = np.arange(1, remaining_from_today, dtype=int)
        # Weights for the advances are linearly decreasing
        weights = np.arange(len(possible_advances), 0, -1, dtype=float)
        # Normalize the weights to get probabilities
        advances_probs = weights / weights.sum()
        # Sample a random advance from the possible advances with the given probabilities
        advance_days = int(self.rng.choice(possible_advances, p=advances_probs))
        # Reservation date
        t_reservation = today + timedelta(days=advance_days)
        # Days that can be booked
        remaining_days_from_reservation = (self.extended_end_simulation - t_reservation).days
        if remaining_days_from_reservation < 0:
            return None  # No days left to stay
        max_stay = int(min(MAX_DAYS_STAY, remaining_days_from_reservation))
        days_stay = int(self.rng.integers(1, max_stay + 1))  # [1, max_stay]
        # Number of guests can be 1, 2 or 3
        num_guests = self.rng.integers(MIN_GUESTS, MAX_GUESTS, endpoint=True)
        return Request(
            t_request=today,
            t_reservation=t_reservation,
            days_stay=days_stay,
            num_guests=num_guests,
        )


# ============ TEST ============
if __name__ == "__main__":
    # ---- Parametri di test ----
    START_SIM = date(2025, 1, 1)
    END_SIM = date(2025, 1, 31)
    W_EXT = 30
    EXT_END_SIM = END_SIM + (timedelta(days=W_EXT))
    SEED = 42
    AVG_REQ = 12

    # Stagionalità fittizia (tutti 1.0)
    seasonality = np.ones(365, dtype=float)

    rg = RequestGenerator(
        avg_num_requests=AVG_REQ,
        start_simulation=START_SIM,
        end_simulation=END_SIM,
        extended_end_simulation=EXT_END_SIM,
        seasonality_func=seasonality,
        seed=SEED,
        precompute=True,
    )

    def validate_request(r: Request, today: date) -> None:
        # campi presenti
        assert hasattr(r, "t_request")
        assert hasattr(r, "t_reservation")

        # t_request è il giorno 'today' (e per definizione < END_SIM)
        assert r.t_request == today

        # check-in (t_reservation) entro l'orizzonte esteso
        assert today < r.t_reservation < EXT_END_SIM, (
            "t_reservation deve essere > today e < extended_end_simulation"
        )

        # soggiorno: deve concludersi entro EXT_END_SIM
        remaining_after_res = (EXT_END_SIM - r.t_reservation).days
        assert 1 <= r.days_stay <= min(MAX_DAYS_STAY, remaining_after_res)

        # ospiti (NumPy: upper bound escluso)
        assert MIN_GUESTS <= r.num_guests < MAX_GUESTS

    def check_requests(today: date) -> np.ndarray:
        """Legge e valida le richieste precompute per 'today'."""
        arr = rg.fetch_requests_for_date(today)
        print(f"[{today}] generated {len(arr)} requests")
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == object
        for r in arr:
            validate_request(r, today)
        return arr

    print("=== RequestGenerator self-test (extended horizon) ===")
    print(f"start={START_SIM}  end={END_SIM}  ext_end={EXT_END_SIM}  avg={AVG_REQ}  seed={SEED}")

    # ---- 1) A ‘normal’ day in the middle of the period ----
    _ = check_requests(date(2025, 1, 10))

    # ---- 2) END_SIM day - 2 (may have requests with check-in after END_SIM) ----
    _ = check_requests(END_SIM - timedelta(days=2))

    # ---- 3) Day END_SIM - 1 (ORA may have requests) ----
    arr_last_useful = rg.fetch_requests_for_date(END_SIM - timedelta(days=1))
    print(
        f"[{END_SIM - timedelta(days=1)}] generated {len(arr_last_useful)} requests (può essere > 0)"
    )
    assert isinstance(arr_last_useful, np.ndarray)
    assert arr_last_useful.dtype == object
    for r in arr_last_useful:
        validate_request(r, END_SIM - timedelta(days=1))

    # ---- 4) Day = END_SIM (out of range for t_request) → always 0 ----
    arr_end = rg.fetch_requests_for_date(END_SIM)
    print(f"[{END_SIM}] generated {len(arr_end)} requests (atteso 0)")
    assert isinstance(arr_end, np.ndarray)
    assert arr_end.dtype == object
    assert len(arr_end) == 0

    total_dict = sum(len(v) for v in rg.requests_by_day.values())
    print(f"Total requests (dict) = {total_dict}")
    print("Self-test completed successfully ✅")
