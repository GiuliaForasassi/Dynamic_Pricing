import pickle
import random
import struct
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from numpy.random import Generator, default_rng
from scipy.stats import norm

from pricing_advertising.exceptions.exceptions import InvalidReservationWindowError
from pricing_advertising.requests.request import Request
from pricing_advertising.utils.constants import (
    HIGH_SEASON,
    MAX_ADVANCE_DAYS,
    MAX_DAYS_STAY,
    MAX_GUESTS,
    MIN_ADVANCE_DAYS,
    MIN_GUESTS,
    MIN_NUM_REQUESTS_FACTOR,
    NUM_WINDOWS,
    REQUESTS_HIGH_SEASON_MULTIPLIER,
    SEEDS,
    STD_FACTOR,
    WINDOW_SIZE,
)


# ----------- REQUESTS GENERATOR --------------
class RequestGenerator:
    def __init__(
        self,
        avg_num_requests: int,  # average number of requests
        start_simulation: date,  # date of starting simulation
        end_simulation: date,  ## date of ending simulation
        extended_end_simulation: date,  # date of ending extended simulation period
        seasonality_array: np.ndarray,  # array of seasonality
        *,
        # Rooms per category (1, 2, 3 guests); used as sampling weights for the number of guests
        num_category1_rooms: int,
        num_category2_rooms: int,
        num_category3_rooms: int,
        seed: int | None = None,
        # If true, all requests for the entire simulation period are precomputed during initialization.
        # If false (on-demand generation), requests are generated as needed for each day (old logic)
        precompute: bool = True,  # If true, precompute all requests at the start
    ) -> None:
        # Checks
        if not isinstance(end_simulation, date) or not isinstance(start_simulation, date):
            raise ValueError("The start_simulation and the end_simulation must be date objects")

        if avg_num_requests < 0:
            raise ValueError("The average number of requests must be greater than 0")

        delta_main = (end_simulation - start_simulation).days
        delta_extended = (extended_end_simulation - start_simulation).days

        if delta_main < 0 or delta_extended <= delta_main:
            raise ValueError(
                "Must be valid: start_simulation <= end_simulation < extendend_end_simulation"
            )

        if len(seasonality_array) != delta_extended:
            raise ValueError(
                "seasonality_array must cover exactly[start_simulation, extended_end_simulation]."
            )

        room_counts = np.array(
            [num_category1_rooms, num_category2_rooms, num_category3_rooms], dtype=float
        )
        if (room_counts < 0).any() or room_counts.sum() <= 0:
            raise ValueError("Room counts must be non-negative with a positive total.")
        self.guests_probs = room_counts / room_counts.sum()

        self.seed = seed
        self.rng: Generator = default_rng(seed)  # NumPy random number generator
        self.py_rng = random.Random(seed)  # Python's built-in random number generator
        self.avg_num_requests = int(avg_num_requests)
        self.start_simulation = start_simulation
        self.end_simulation = end_simulation
        self.extended_end_simulation = extended_end_simulation
        self.seasonality_array = seasonality_array

        # Dictionary to store precomputed requests by day
        self.requests_by_day: dict[date, list[Request]] = {}
        if precompute:
            self._pre_generate_all_requests()

    # ---------------- Helper functions -------------------------
    # Return the requests for the given date as a NumPy array
    def fetch_requests_for_date(self, today: date) -> np.ndarray:
        return np.array(self.requests_by_day.get(today, []), dtype=object)

    def ensure_precomputed(self) -> None:
        """Ensure that all requests have been precomputed."""
        if not getattr(self, "requests_by_day", {}):
            self._pre_generate_all_requests()

    # Permit regenerating the requests with a new seed
    def regenerate(self, seed: int | None = None) -> None:
        """Generate the next calendar, or replay a calendar with an explicit seed."""
        if seed is not None:
            # Update the seed
            self.seed = seed
            # Re-seed the random number generators
            self.rng = default_rng(self.seed)
            self.py_rng = random.Random(self.seed)
        self.requests_by_day.clear()
        self._pre_generate_all_requests()

    # --------------- Pre-generation function -------------
    # Pre-generate all requests for the entire simulation period
    def _pre_generate_all_requests(self) -> None:
        # Take the date of the starting simulation period
        current = self.start_simulation
        # Loop through each day in the simulation period
        while current <= self.end_simulation:
            # Sample the number of requests for the current day
            num_requests = self._sample_num_requests(current)
            # Initialize an empty list to store the requests for the current day
            requests: list[Request] = []
            for _ in range(num_requests):
                # Generate a new request for the current day
                request = self._generate_request(current)
                if request is not None:
                    requests.append(request)
            # Store only if there are requests
            if requests:
                self.requests_by_day[current] = requests
            current += timedelta(days=1)

    # ------------- Sampling number of requests function -----------
    # Sample the number of requests for a given day combining seasonality, average number of requests, and stochastic distribution
    def _sample_num_requests(self, today: date) -> int:
        # Set the initial seasonality factor
        # Update the seasonality factor if today is in the high season
        # Adjust the mean number of requests based on the seasonality factor
        adjusted_mean = max(0.0, self.avg_num_requests)
        # Create a normal distribution with the adjusted mean and standard deviation
        distribution = norm(
            loc=adjusted_mean, scale=max(1e-6, STD_FACTOR * max(1.0, adjusted_mean))
        )
        # Sample from the distribution
        sample = distribution.rvs(size=1, random_state=self.rng)[0]
        # Calculate the minimum number of requests based on the average number of requests and a predefined factor
        min_num_requests = int(max(0, MIN_NUM_REQUESTS_FACTOR * self.avg_num_requests))
        # Ensure the number of requests is an integer and non-negative
        num_requests = int(np.round(max(sample, min_num_requests)))
        # Return the final number of requests
        return num_requests

    # ---------------- Generation request function -----------------
    def _generate_request(self, today: date) -> Request | None:
        booking_end = min(
            self.extended_end_simulation,
            today + timedelta(days=NUM_WINDOWS * WINDOW_SIZE + 1),
        )
        remaining_from_today = (booking_end - today).days
        if remaining_from_today <= 1:
            return None  # No more requests can be made

        # Possible advances are between 1 and maximum_advance - 1
        possible_advances = np.arange(MIN_ADVANCE_DAYS, MAX_ADVANCE_DAYS, dtype=int)
        # Weights for the advances are linearly decreasing
        time_weights = np.arange(len(possible_advances), 0, -1, dtype=float)
        # Seasonality weights
        start_idx = (today - self.start_simulation).days + 1
        seasonality_weights = (
            self.seasonality_array[start_idx : start_idx + len(possible_advances)] + 1
        )
        weights = time_weights * seasonality_weights
        # Normalize the weights to get probabilities
        advances_probs = weights / weights.sum()
        # Sample a random advance from the possible advances with the given probabilities
        advance_days = int(self.rng.choice(possible_advances, p=advances_probs))
        # Reservation date
        t_reservation = today + timedelta(days=advance_days)
        # Days that can be booked
        remaining_days_from_reservation = (booking_end - t_reservation).days
        if remaining_days_from_reservation < 0:
            return None  # No days left to stay
        max_stay = int(min(MAX_DAYS_STAY, remaining_days_from_reservation))
        days_stay = int(self.rng.integers(1, max_stay + 1))  # [1, max_stay]
        # Number of guests (1, 2 or 3) weighted by the rooms available per category
        num_guests = int(
            self.rng.choice(np.arange(MIN_GUESTS, MAX_GUESTS + 1), p=self.guests_probs)
        )
        return Request(
            t_request=today,
            t_reservation=t_reservation,
            days_stay=days_stay,
            num_guests=num_guests,
        )

    # ----------------- Optimization functions -----------------
    # It removes heavy or non-serializable objects
    # and compresses the entire dictionary of daily requests into a compact binary format using the struct module
    def __getstate__(self) -> dict:
        """Write version 2 request records (12 bytes each), preserving RNG state."""
        state = self.__dict__.copy()
        packed_chunks = []
        req_struct = struct.Struct("<iihh")

        for day_key, req_list in self.requests_by_day.items():
            day_ord = day_key.toordinal()
            count = len(req_list)
            # Header for each day entry: ordinal date + request count
            packed_chunks.append(struct.pack("<ii", day_ord, count))

            for request in req_list:
                packed_chunks.append(
                    req_struct.pack(
                        request.t_request.toordinal(),
                        request.t_reservation.toordinal(),
                        request.days_stay,
                        request.num_guests,
                    )
                )

        state["requests_format_version"] = 2
        state["requests_by_day_bytes"] = b"".join(packed_chunks)
        # Exclude the original slow dictionary from pickle's work
        state["requests_by_day"] = {}

        return state

    # It reads the stream of saved binary bytes, efficiently reconstructs the request dictionary,
    # and reinitializes the statistical and random number generators
    def __setstate__(self, state: dict) -> None:
        """Read current or unversioned legacy records, discarding old proposal fields."""
        state = state.copy()
        version = state.pop("requests_format_version", 1)
        if version not in (1, 2):
            raise ValueError(f"Unsupported request serialization version: {version}")
        raw_bytes = state.pop("requests_by_day_bytes", b"")
        state.pop("base_dist", None)
        self.__dict__.update(state)

        # 1. Reconstruct requests_by_day fast from raw bytes
        requests_by_day = {}
        if raw_bytes:
            header_struct = struct.Struct("<ii")
            req_struct = struct.Struct("<iihhih" if version == 1 else "<iihh")
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
                    t_req, t_res, stay, guests = req_struct.unpack_from(raw_bytes, offset)[:4]
                    offset += req_size

                    req = Request(
                        t_request=date.fromordinal(t_req),
                        t_reservation=date.fromordinal(t_res),
                        days_stay=stay,
                        num_guests=guests,
                    )
                    req_list.append(req)

                requests_by_day[day_key] = req_list

        self.requests_by_day = requests_by_day

        if "rng" not in state:
            self.rng = default_rng(self.seed)
        if "py_rng" not in state:
            self.py_rng = random.Random(self.seed)


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
        num_category1_rooms=100,
        num_category2_rooms=50,
        num_category3_rooms=30,
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
