import pickle
import random
import struct
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from numpy.random import Generator, default_rng
from scipy.stats import norm, rv_continuous

from pricing_advertising.exceptions.exceptions import InvalidReservationWindowError
from pricing_advertising.requests.request import Request
from pricing_advertising.utils.constants import (
    HIGH_SEASON,
    HIGH_SEASON_MULTIPLIER,
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
        seasonality_array: np.ndarray,
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
        self.seasonality_array = seasonality_array

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
        # Get the seasonality factor for the current day
        seasonality_factor = 1
        if int(today.month in HIGH_SEASON):
            seasonality_factor = HIGH_SEASON_MULTIPLIER
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
        time_weights = np.arange(len(possible_advances), 0, -1, dtype=float)
        # Seasonality weights
        seasonality_weights = self.seasonality_array[(today - self.start_simulation).days + 1 :] + 1
        weights = time_weights * seasonality_weights
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

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()

        # 1. Un-pickable or heavy RNG / SciPy objects are excluded.
        # They will be recreated in __setstate__ using self.seed and parameters.
        state.pop("base_dist", None)
        state.pop("rng", None)
        state.pop("py_rng", None)

        # 2. Fast binary pack for self.requests_by_day
        # Convert date objects and Request attributes into a compact byte format:
        # Date key: 4 bytes (ordinal)
        # Request count: 4 bytes
        # Per Request: 4b (t_req), 4b (t_res), 2b (stay), 2b (guests), 4b (prop_res), 2b (prop_stay)
        packed_chunks = []

        # Binary format for a single request:
        # i = int32 (4 bytes), h = int16 (2 bytes)
        # (t_req, t_res, days_stay, num_guests, proposed_t_res, proposed_days_stay)
        req_struct = struct.Struct("<iihhih")

        for day_key, req_list in self.requests_by_day.items():
            day_ord = day_key.toordinal()
            count = len(req_list)
            # Header for each day entry: ordinal date + request count
            packed_chunks.append(struct.pack("<ii", day_ord, count))

            for r in req_list:
                t_req = r.t_request.toordinal()
                t_res = r.t_reservation.toordinal()
                prop_res = r.proposed_t_reservation.toordinal() if r.proposed_t_reservation else -1
                prop_stay = r.proposed_days_stay if r.proposed_days_stay is not None else -1

                packed_chunks.append(
                    req_struct.pack(t_req, t_res, r.days_stay, r.num_guests, prop_res, prop_stay)
                )

        state["requests_by_day_bytes"] = b"".join(packed_chunks)
        # Exclude the original slow dictionary from pickle's work
        state["requests_by_day"] = {}

        return state

    def __setstate__(self, state: dict) -> None:
        raw_bytes = state.pop("requests_by_day_bytes", b"")
        self.__dict__.update(state)

        # 1. Reconstruct requests_by_day fast from raw bytes
        requests_by_day = {}
        if raw_bytes:
            header_struct = struct.Struct("<ii")
            req_struct = struct.Struct("<iihhih")
            header_size = header_struct.size
            req_size = req_struct.size

            offset = 0
            total_len = len(raw_bytes)

            while offset < total_len:
                day_ord, count = header_struct.unpack_from(raw_bytes, offset)
                offset += header_size
                day_key = date.fromordinal(day_ord)

                req_list = []
                for _ in range(count):
                    t_req, t_res, stay, guests, prop_res, prop_stay = req_struct.unpack_from(
                        raw_bytes, offset
                    )
                    offset += req_size

                    req = Request(
                        t_request=date.fromordinal(t_req),
                        t_reservation=date.fromordinal(t_res),
                        days_stay=stay,
                        num_guests=guests,
                    )
                    req.proposed_t_reservation = (
                        date.fromordinal(prop_res) if prop_res != -1 else None
                    )
                    req.proposed_days_stay = prop_stay if prop_stay != -1 else None
                    req_list.append(req)

                requests_by_day[day_key] = req_list

        self.requests_by_day = requests_by_day

        # 2. Re-initialize RNG and SciPy distribution objects
        self.rng = default_rng(self.seed)
        self.py_rng = random.Random(self.seed)

        # Re-create base distribution if it wasn't customized
        # Note: If custom distribution was passed, pass its parameters via state instead
        STD_FACTOR = 0.2  # Match whatever STD_FACTOR constant you have
        self.base_dist = norm(
            loc=self.avg_num_requests,
            scale=max(1e-6, STD_FACTOR * max(1, self.avg_num_requests)),
        )


# ============ TEST ============
# if __name__ == "__main__":
#     # ---- Parametri di test ----
#     START_SIM = date(2025, 1, 1)
#     END_SIM = date(2025, 1, 31)
#     W_EXT = 30
#     EXT_END_SIM = END_SIM + (timedelta(days=W_EXT))
#     SEED = 42
#     AVG_REQ = 12

#     # Stagionalità fittizia (tutti 1.0)
#     seasonality = np.ones(365, dtype=float)

#     rg = RequestGenerator(
#         avg_num_requests=AVG_REQ,
#         start_simulation=START_SIM,
#         end_simulation=END_SIM,
#         extended_end_simulation=EXT_END_SIM,
#         seasonality_array=seasonality,
#         seed=SEED,
#         precompute=True,
#     )

#     def validate_request(r: Request, today: date) -> None:
#         # campi presenti
#         assert hasattr(r, "t_request")
#         assert hasattr(r, "t_reservation")

#         # t_request è il giorno 'today' (e per definizione < END_SIM)
#         assert r.t_request == today

#         # check-in (t_reservation) entro l'orizzonte esteso
#         assert today < r.t_reservation < EXT_END_SIM, (
#             "t_reservation deve essere > today e < extended_end_simulation"
#         )

#         # soggiorno: deve concludersi entro EXT_END_SIM
#         remaining_after_res = (EXT_END_SIM - r.t_reservation).days
#         assert 1 <= r.days_stay <= min(MAX_DAYS_STAY, remaining_after_res)

#         # ospiti (NumPy: upper bound escluso)
#         assert MIN_GUESTS <= r.num_guests < MAX_GUESTS

#     def check_requests(today: date) -> np.ndarray:
#         """Legge e valida le richieste precompute per 'today'."""
#         arr = rg.fetch_requests_for_date(today)
#         print(f"[{today}] generated {len(arr)} requests")
#         assert isinstance(arr, np.ndarray)
#         assert arr.dtype == object
#         for r in arr:
#             validate_request(r, today)
#         return arr

#     print("=== RequestGenerator self-test (extended horizon) ===")
#     print(f"start={START_SIM}  end={END_SIM}  ext_end={EXT_END_SIM}  avg={AVG_REQ}  seed={SEED}")

#     # ---- 1) A ‘normal’ day in the middle of the period ----
#     _ = check_requests(date(2025, 1, 10))

#     # ---- 2) END_SIM day - 2 (may have requests with check-in after END_SIM) ----
#     _ = check_requests(END_SIM - timedelta(days=2))

#     # ---- 3) Day END_SIM - 1 (ORA may have requests) ----
#     arr_last_useful = rg.fetch_requests_for_date(END_SIM - timedelta(days=1))
#     print(
#         f"[{END_SIM - timedelta(days=1)}] generated {len(arr_last_useful)} requests (può essere > 0)"
#     )
#     assert isinstance(arr_last_useful, np.ndarray)
#     assert arr_last_useful.dtype == object
#     for r in arr_last_useful:
#         validate_request(r, END_SIM - timedelta(days=1))

#     # ---- 4) Day = END_SIM (out of range for t_request) → always 0 ----
#     arr_end = rg.fetch_requests_for_date(END_SIM)
#     print(f"[{END_SIM}] generated {len(arr_end)} requests (atteso 0)")
#     assert isinstance(arr_end, np.ndarray)
#     assert arr_end.dtype == object
#     assert len(arr_end) == 0

#     total_dict = sum(len(v) for v in rg.requests_by_day.values())
#     print(f"Total requests (dict) = {total_dict}")
#     print("Self-test completed successfully ✅")

# if __name__ == "__main__":
#     import pickle
#     from time import perf_counter

#     start = date(2026, 1, 1)
#     end = date(2026, 12, 31)
#     ext_end = date(2027, 1, 15)
#     seasonality = np.ones((ext_end - start).days, dtype=np.float64)

#     generator_instance = RequestGenerator(
#         avg_num_requests=1000,
#         start_simulation=start,
#         end_simulation=end,
#         extended_end_simulation=ext_end,
#         seasonality_array=seasonality,
#         distribution=None,  # Pass custom scipy distribution object if needed
#         seed=42,
#         precompute=True,
#     )

#     # --- Step 2: Measure Pickle Serialization Time ---
#     start_time = perf_counter()

#     serialized_bytes = pickle.dumps(generator_instance)

#     end_time = perf_counter()

#     # --- Step 3: Output Results ---
#     elapsed_time_ms = (end_time - start_time) * 1000
#     payload_size_mb = len(serialized_bytes) / (1024 * 1024)

#     print(f"Serialization successful!")
#     print(f"Time taken: {elapsed_time_ms:.4f} ms")
#     print(f"Pickled size: {payload_size_mb:.2f} MB")


if __name__ == "__main__":
    import pickle
    from time import perf_counter

    start = date(2026, 1, 1)
    end = date(2026, 12, 31)
    ext_end = date(2027, 1, 15)
    seasonality = np.ones((ext_end - start).days, dtype=np.float64)

    generator_instance = RequestGenerator(
        avg_num_requests=1000,
        start_simulation=start,
        end_simulation=end,
        extended_end_simulation=ext_end,
        seasonality_array=seasonality,
        distribution=None,
        seed=42,
        precompute=True,
    )

    # --- Test Serialization ---
    start_time = perf_counter()
    serialized_bytes = pickle.dumps(generator_instance)
    end_time = perf_counter()

    elapsed_time_ms = (end_time - start_time) * 1000
    payload_size_mb = len(serialized_bytes) / (1024 * 1024)

    print("=== SERIALIZATION SUCCESSFUL ===")
    print(f"Time taken: {elapsed_time_ms:.2f} ms")
    print(f"Pickled size: {payload_size_mb:.2f} MB")

    # --- Test Deserialization ---
    start_time = perf_counter()
    restored_instance = pickle.loads(serialized_bytes)
    end_time = perf_counter()

    print(f"Deserialization time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Restored days count: {len(restored_instance.requests_by_day)}")
