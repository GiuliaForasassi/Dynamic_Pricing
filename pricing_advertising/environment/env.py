import random
from calendar import isleap
from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
from typing import Any, cast

import gymnasium as gym
import numpy as np
from loguru import logger

from pricing_advertising.acceptance.acceptance import ABCAcceptanceModel
from pricing_advertising.acceptance.logistic_function import LogisticAcceptanceModel
from pricing_advertising.exceptions.exceptions import (
    InvalidRoomCategoryError,
    NotValidPeriodError,
)
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.hotel.room import ROOM_ALTERNATIVES, RoomCategory
from pricing_advertising.pricing_model.bar_table import get_price, n_levels
from pricing_advertising.pricing_model.bernstein_model import BernsteinProbModel
from pricing_advertising.requests.request import HotelOffer, Request
from pricing_advertising.requests.request_generator import RequestGenerator
from pricing_advertising.utils.constants import (
    HIGH_SEASON,
    LOGISTIC_ACCEPTANCE_AVG_PRICES,
    LOGISTIC_ACCEPTANCE_BETA,
    LOGISTIC_ACCEPTANCE_HIGH_SEASON_MULTIPLIERS,
    LOGISTIC_ACCEPTANCE_STDS,
    MAX_AVG_NUM_REQUESTS,
    MIN_AVG_NUM_REQUESTS,
    NUM_COEFFICIENTS,
    NUM_WINDOWS,
    WINDOW_SIZE,
)


# -------------------- ENVIRONMENT --------------------
class PricingEnv(gym.Env):
    def __init__(
        self,
        start_simulation: date,  # The start date of the simulation
        end_simulation: date,  # The end date of the simulation
        hotel: Hotel,  # The hotel instance for the simulation
        avg_num_requests: int | None = None,  # The average number of requests per day
        request_generator: RequestGenerator | None = None,  # The request generator instance
        acceptance_model: ABCAcceptanceModel
        | None = None,  # The acceptance model for booking requests
        seed: int | None = None,  # The random seed for reproducibility
        collect_eval_artifacts: bool = False,  # Whether to collect evaluation artifacts
        artifact_snapshot_dates: list[date]
        | None = None,  # The dates for which to collect artifacts
        allow_category_alternative: bool = True,
        allow_period_alternative: bool = True,
    ) -> None:
        # Validate the simulation period
        if start_simulation > end_simulation:
            raise ValueError(
                f"The start date {start_simulation} must be before the end date {end_simulation}."
            )

        super().__init__()

        # -------- Random and seed setup --------
        self.seed = seed
        self._init_dynamics_rngs(seed)
        self.py_rng = random.Random(seed)

        # ------------ Simulation horizon and hotel ------------
        self.start_simulation = start_simulation
        self.end_simulation = end_simulation
        self.total_days = (end_simulation - start_simulation).days + 1

        self.window_span_days = (
            NUM_WINDOWS * WINDOW_SIZE
        )  # Total number of days covered by the observation windows
        self.extended_simulation_days = (
            self.total_days + self.window_span_days
        )  # Total number of days including the observation window span

        # Is needed because a client can book today for a future day that is beyond the original simulation horizon
        self.extended_end_simulation = start_simulation + timedelta(
            days=self.extended_simulation_days
        )

        self.hotel = hotel
        self.num_room_categories = len(RoomCategory)
        if sum(self.hotel.get_num_rooms_per_category(category) for category in RoomCategory) <= 0:
            raise ValueError("Total hotel capacity must be positive.")
        self.allow_category_alternative = allow_category_alternative
        self.allow_period_alternative = allow_period_alternative

        self.collect_eval_artifacts = collect_eval_artifacts
        self.artifact_snapshot_dates = set(artifact_snapshot_dates or [])

        # ----------- State variables -----------
        # Initialize the current day of the simulation
        self.today = start_simulation
        # Initialize the BAR table levels
        self.n_levels = n_levels()
        if self.n_levels <= 0:
            raise RuntimeError("BAR table not initialised: n_levels()=0")
        if self.n_levels < 2:
            raise ValueError("BAR table must contain the initial sellable level 1.")
        self._max_bar_level = self.n_levels - 1
        # Initialize occupancy, demand, and BAR levels arrays
        self.occupancy = np.zeros(
            (self.num_room_categories, self.extended_simulation_days), dtype=int
        )
        self.demand = np.zeros((self.num_room_categories, self.extended_simulation_days), dtype=int)
        self.bar_levels = np.ones(
            (self.num_room_categories, self.extended_simulation_days), dtype=int
        )

        # ----------- Observation space -----------
        self.observation_space = gym.spaces.Dict(
            {
                "occupancy": gym.spaces.Box(
                    low=0, high=1, shape=(self.num_room_categories, NUM_WINDOWS), dtype=np.float32
                ),
                "demand": gym.spaces.Box(
                    low=0, high=1, shape=(self.num_room_categories, NUM_WINDOWS), dtype=np.float32
                ),
                # Represent a day as a integer number
                "today": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                # Seasonality is represented as a discrete value: 0 for low season, 1 for high season
                "seasonality": gym.spaces.Discrete(2, seed=seed),
                "calendar": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                # Profile of BARs by category and by window --> represents where we are (current status)
                "bar_levels": gym.spaces.Box(
                    low=0, high=1, shape=(self.num_room_categories, NUM_WINDOWS), dtype=np.float32
                ),
            }
        )

        # ----------- Action space -----------
        # Action space is coefficients of the Bernstein polynomial
        # Vector of coefficients for the Bernstein polynomial for each room category
        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.num_room_categories, 2, NUM_COEFFICIENTS),
            dtype=np.float32,
        )

        # ----------- Request generator and acceptance model -----------
        # Determine the average number of requests per day
        if avg_num_requests is None:
            avg_num_requests = (
                request_generator.avg_num_requests
                if request_generator is not None
                else self.py_rng.randint(MIN_AVG_NUM_REQUESTS, MAX_AVG_NUM_REQUESTS)
            )
        if (
            isinstance(avg_num_requests, bool)
            or not isinstance(avg_num_requests, int | np.integer)
            or avg_num_requests < 0
        ):
            raise ValueError("The average number of requests must be a non-negative integer.")
        self.avg_num_requests = int(avg_num_requests)
        # Compute the seasonality array for the extended simulation period
        self.seasonality_array = np.array(
            [
                self.get_seasonality(start_simulation + timedelta(days=day_idx))
                for day_idx in range(self.extended_simulation_days)
            ]
        )

        # Instantiate the request generator and acceptance model
        if request_generator is not None and (
            request_generator.start_simulation > start_simulation
            or request_generator.end_simulation < end_simulation
            or request_generator.extended_end_simulation < self.extended_end_simulation
        ):
            raise ValueError(
                "The request generator must cover the simulation period and extended booking horizon."
            )
        self.request_generator = request_generator or RequestGenerator(
            self.avg_num_requests,
            start_simulation,
            end_simulation,
            extended_end_simulation=self.extended_end_simulation,
            seasonality_array=self.seasonality_array,
            seed=seed,
            precompute=True,  # Always precompute requests at the beginning
        )

        if request_generator is not None:
            self.request_generator.ensure_precomputed()

        # Logistic acceptance model: the probability that a client accepts depends on how much the offered price deviates from the average expected price
        self.acceptance_model = acceptance_model or LogisticAcceptanceModel(
            beta=LOGISTIC_ACCEPTANCE_BETA,
            avg_prices=LOGISTIC_ACCEPTANCE_AVG_PRICES,
            st_deviations=LOGISTIC_ACCEPTANCE_STDS,
            high_season_multipliers=LOGISTIC_ACCEPTANCE_HIGH_SEASON_MULTIPLIERS,
        )
        self.acceptance_model.rng = self.acceptance_rng

        # --------- Variables for tracking requests and prices ---------
        # Store final requests and prices for tracker access
        self.last_total_requests: np.ndarray = np.array([])
        self.last_valid_requests: np.ndarray = np.array([])
        self.last_valid_prices: np.ndarray = np.array([])
        self.last_final_requests: np.ndarray = np.array([])
        self.last_final_prices: np.ndarray = np.array([])

        # Attribute for counting accepted/rejected requests
        self.accepted_by_day = np.zeros(
            (self.num_room_categories, self.extended_simulation_days), dtype=np.int32
        )
        self.rejected_by_day = np.zeros_like(self.accepted_by_day)

        self.accepted_total = 0
        self.rejected_total = 0
        self.last_day_avg_price = 0.0

    # ---------- Initialize dynamics RNGs ---------------
    def _init_dynamics_rngs(self, seed: int | None) -> None:
        bar_seed, alternative_seed, acceptance_seed = np.random.SeedSequence(seed).spawn(3)
        self.np_rng = np.random.default_rng(bar_seed)
        self.alternative_rng = np.random.default_rng(alternative_seed)
        self.acceptance_rng = np.random.default_rng(acceptance_seed)

    # ---------- Get observation ---------------
    # Returns the current observation of the environment, including occupancy, demand, bar levels, and seasonality information
    # It is what the agent observes at the current time step
    def get_observation(self) -> dict:
        current_day_index = (self.today - self.start_simulation).days
        simulation_days = self.total_days

        occupancy_windows = np.zeros((self.num_room_categories, NUM_WINDOWS), dtype=np.float32)
        demand_windows = np.zeros((self.num_room_categories, NUM_WINDOWS), dtype=np.float32)
        bar_levels_windows = np.zeros((self.num_room_categories, NUM_WINDOWS), dtype=np.float32)

        # The environment aims to create an “aggregate vision” of the near future for each room category
        for category in range(self.num_room_categories):
            num_rooms = self.hotel.get_num_rooms_per_category(RoomCategory(category + 1))

            for window in range(NUM_WINDOWS):
                # Calculate the start and end indices for the current window
                start = current_day_index + 1 + window * WINDOW_SIZE
                end = min(start + WINDOW_SIZE, self.extended_simulation_days)
                width = end - start
                if width <= 0:
                    continue

                # Calculate the occupancy and demand sums for the current window in order to have a aggregate vision
                occupancy_sum = np.sum(self.occupancy[category, start:end])
                demand_sum = np.sum(self.demand[category, start:end])

                # Normalize the occupancy and demand values
                max_occupancy = max(1, num_rooms * width)
                max_demand = max(1, width * self.avg_num_requests)
                occupancy_norm = occupancy_sum / max_occupancy
                demand_norm = demand_sum / max_demand

                occupancy_windows[category, window] = np.float32(occupancy_norm)
                demand_windows[category, window] = np.float32(demand_norm)

                # Bar levels
                bl = self.bar_levels[category, start:end]
                if bl.size > 0 and self._max_bar_level > 0:
                    bar_levels_windows[category, window] = np.float32(
                        bl.mean() / self._max_bar_level
                    )
                else:
                    bar_levels_windows[category, window] = np.float32(0.0)

        today_norm = np.array(
            [min(1.0, max(0.0, current_day_index / max(1, simulation_days)))],
            dtype=np.float32,
        )
        seasonality = int(self.get_seasonality(self.today))
        day_of_year = (self.today - date(self.today.year, 1, 1)).days
        days_in_year = 366 if isleap(self.today.year) else 365
        phase = 2 * np.pi * day_of_year / days_in_year
        calendar = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)

        np.clip(occupancy_windows, 0.0, 1.0, out=occupancy_windows)
        np.clip(demand_windows, 0.0, 1.0, out=demand_windows)
        np.clip(bar_levels_windows, 0.0, 1.0, out=bar_levels_windows)

        return {
            "occupancy": occupancy_windows,
            "demand": demand_windows,
            "today": today_norm,
            "seasonality": seasonality,
            "calendar": calendar,
            "bar_levels": bar_levels_windows,
        }

    # ------------ Seasonality and stay indices ---------------
    def get_seasonality(self, time: date) -> int:
        return int(time.month in HIGH_SEASON)

    def _booking_window_indices(self) -> tuple[int, int]:
        start = (self.today - self.start_simulation).days + 1
        return start, min(start + self.window_span_days, self.extended_simulation_days)

    def _within_booking_window(self, offer: Request | HotelOffer) -> bool:
        start, stop = self._booking_window_indices()
        first = (offer.t_reservation - self.start_simulation).days
        return offer.days_stay > 0 and start <= first and first + offer.days_stay <= stop

    # Get the indices for the stay period of a request, including category, start, and end indices
    def _stay_indices(self, req: Request | HotelOffer) -> tuple[int, int, int]:
        cat_idx = req.category.value - 1
        start_idx = (req.t_reservation - self.start_simulation).days
        end_idx = start_idx + req.days_stay

        if cat_idx < 0 or cat_idx >= self.num_room_categories:
            raise InvalidRoomCategoryError(req.category.value)
        if req.days_stay <= 0 or start_idx < 0 or end_idx > self.extended_simulation_days:
            raise NotValidPeriodError(
                start_period=req.t_reservation,
                end_period=req.t_reservation + timedelta(days=req.days_stay),
                start_simulation=self.start_simulation,
                end_simulation=self.start_simulation
                + timedelta(days=self.extended_simulation_days),
            )
        return cat_idx, start_idx, end_idx

    # ---------------- STEP Method ------------------
    # Perform a simulation step for the current day, including request generation, price evaluation, and occupancy update.
    def step(self, action) -> tuple:  # noqa: ANN001
        if self.today > self.end_simulation:
            raise gym.error.ResetNeeded("Episode has terminated; call reset() before step().")
        action = self._validate_action(action)
        # Generate requests for the current day
        requests = self.request_generator.fetch_requests_for_date(self.today)

        # Check if the requests can be accommodated
        valid_mask, valid_requests = self._process_requests(requests, self.today)

        # Update the demand
        self._update_demand(requests)

        # Snapshot the bar levels before processing today's requests
        # If you're collecting evaluation artifacts and today is a snapshot date, save a copy of the bar_levels before they are updated.
        # This is only needed for the “before/after” chart.
        bar_before = None
        if self.collect_eval_artifacts and self.today in self.artifact_snapshot_dates:
            snapshot_start = (self.today - self.start_simulation).days
            bar_before = self.bar_levels[:, snapshot_start : self.total_days].tolist()

        bar_history = {}
        if self.collect_eval_artifacts:
            for stay_date in sorted(self.artifact_snapshot_dates):
                if self.today <= stay_date <= self.end_simulation:
                    stay_index = (stay_date - self.start_simulation).days
                    bar_history[stay_date.isoformat()] = {
                        "levels_before": self.bar_levels[:, stay_index].tolist(),
                        "occupancy_before": self.occupancy[:, stay_index].tolist(),
                    }

        # Generate prices for each day of valid requests
        daily_prices = self._generate_prices(valid_requests, action)

        # Accept or reject the requests
        (accepted_mask, tot_prices) = self._evaluate_prices(daily_prices, valid_requests)
        confirmed_mask, offer_attempts = self._confirm_offers(
            valid_requests, daily_prices, accepted_mask, tot_prices
        )
        final_requests = self._apply_mask(confirmed_mask, valid_requests)
        final_prices = self._apply_mask(confirmed_mask, tot_prices)

        # --------- Update accepted and rejected requests by day ---------

        current_day_index = (self.today - self.start_simulation).days

        # tot richieste processabili oggi (dopo availability)
        tot_per_cat = Counter(r.category.value - 1 for r in valid_requests)
        # effettivamente confermate (dopo acceptance + occupancy)
        acc_per_cat = Counter(r.category.value - 1 for r in final_requests)

        if 0 <= current_day_index < self.extended_simulation_days:
            for c in range(self.num_room_categories):
                accepted_c = int(acc_per_cat.get(c, 0))
                rejected_c = max(int(tot_per_cat.get(c, 0) - accepted_c), 0)

                self.accepted_by_day[c, current_day_index] += accepted_c
                self.rejected_by_day[c, current_day_index] += rejected_c
                self.accepted_total += accepted_c
                self.rejected_total += rejected_c

        # Calculate the revenue of accepted requests
        reward = self._calculate_reward(final_prices)

        # Store for tracker access
        self.last_total_requests = requests
        self.last_valid_requests = valid_requests
        self.last_valid_prices = tot_prices
        self.last_final_requests = final_requests
        self.last_final_prices = final_prices

        info: dict[str, Any] = {}

        # For logging metrics
        # Dati dello step appena elaborato, prima dell'incremento
        # della data e dell'eventuale reset automatico.
        total_rooms = sum(
            self.hotel.get_num_rooms_per_category(RoomCategory(c + 1))
            for c in range(self.num_room_categories)
        )
        if total_rooms <= 0:
            raise ValueError("La capacità totale dell'hotel deve essere positiva.")

        confirmed_valid_indices = np.flatnonzero(confirmed_mask)

        # Ordine: richiesta confermata, poi notti di quella richiesta.
        final_nightly_prices = np.asarray(
            [
                float(price)
                for index in confirmed_valid_indices
                for price in daily_prices[int(index)]
            ],
            dtype=float,
        )

        if len(final_nightly_prices) != sum(int(request.days_stay) for request in final_requests):
            raise ValueError("Prezzi notturni non allineati alle richieste confermate.")

        reservation_offsets = [
            (request.t_reservation - self.start_simulation).days for request in requests
        ]

        # --------- Store step metrics for logging ---------
        info["step_metrics"] = {
            "day": self.today,
            "day_index": int(current_day_index),
            "start_simulation": self.start_simulation,
            "simulation_days": self.total_days,
            "occupancy_pct": (
                float(self.occupancy[:, current_day_index].sum()) / total_rooms * 100.0
            ),
            "num_requests": len(requests),
            # Tutte le richieste, non soltanto quelle confermate.
            # Ogni coppia contiene: indice del check-in, numero di notti.
            "request_stays": [
                (
                    int((request.t_reservation - self.start_simulation).days),
                    int(request.days_stay),
                )
                for request in requests
            ],
            # Manteniamo temporaneamente la convenzione del tracker attuale.
            "mean_check_in_offset": (
                float(np.mean(reservation_offsets)) if reservation_offsets else 0.0
            ),
            "reward": float(reward),
            "num_offers": len(valid_requests),
            "num_price_accepted": int(np.count_nonzero(accepted_mask)),
            "num_confirmed": len(final_requests),
            # Copie indipendenti dallo stato mutabile dell'ambiente.
            "confirmed_requests": deepcopy(final_requests),
            "confirmed_nightly_prices": final_nightly_prices,
        }

        # For plottings artifacts
        if self.collect_eval_artifacts:
            request_rows = []

            for request_id, request in enumerate(requests):
                row = self._request_to_artifact(request)
                row["requested"] = row.copy()
                row.update(
                    {
                        "request_id": request_id,
                        "valid": False,
                        "price_accepted": False,
                        "confirmed": False,
                        "daily_prices": None,
                    }
                )
                request_rows.append(row)
            valid_positions = np.flatnonzero(np.asarray(valid_mask, dtype=bool))

            for position, offer, prices, price_accepted, confirmed, attempts in zip(
                valid_positions,
                valid_requests,
                daily_prices,
                accepted_mask,
                confirmed_mask,
                offer_attempts,
                strict=True,
            ):
                request_rows[int(position)].update(
                    {
                        "check_in": offer.t_reservation.isoformat(),
                        "days_stay": int(offer.days_stay),
                        "category": int(offer.category.value),
                        "period_changed": offer.period_changed,
                        "valid": True,
                        "price_accepted": bool(price_accepted),
                        "confirmed": bool(confirmed),
                        "daily_prices": [float(price) for price in prices],
                        "offer_attempts": attempts,
                    }
                )

            payload: dict[str, Any] = {
                "request_day": self.today.isoformat(),
                "day_index": int((self.today - self.start_simulation).days),
                "start_simulation": self.start_simulation.isoformat(),
                "end_simulation": self.end_simulation.isoformat(),
                "step_reward": float(reward),
                "requests": request_rows,
                "policy_coefficients": np.asarray(action, dtype=float).tolist(),
            }

            for stay_day, point in bar_history.items():
                stay_index = (date.fromisoformat(stay_day) - self.start_simulation).days
                point["levels_after"] = self.bar_levels[:, stay_index].tolist()
                point["occupancy_after"] = self.occupancy[:, stay_index].tolist()
            payload["bar_history"] = bar_history
            payload["max_bar_level"] = int(self._max_bar_level)
            _, booking_stop = self._booking_window_indices()
            payload["bar_update_end_day"] = (
                self.start_simulation + timedelta(days=booking_stop)
            ).isoformat()

            if self.today in self.artifact_snapshot_dates:
                start = (self.today - self.start_simulation).days
                stop = (self.end_simulation - self.start_simulation).days + 1

                levels = self.bar_levels[:, start:stop]

                payload["snapshot_start_day"] = self.today.isoformat()
                payload["seasonality_by_day"] = self.seasonality_array[start:stop].tolist()
                payload["occupancy_by_category"] = self.occupancy[:, start:stop].tolist()
                payload["bar_levels_by_category"] = levels.tolist()
                assert bar_before is not None
                payload["bar_levels_before_update"] = bar_before

                # Salviamo direttamente le tariffe: i grafici non dipenderanno
                # dalla tabella BAR caricata quando verranno rigenerati.
                payload["bar_prices_by_category"] = [
                    [float(get_price(RoomCategory(c + 1).name, int(level))) for level in levels[c]]
                    for c in range(self.num_room_categories)
                ]

                payload["rooms_by_category"] = [
                    int(self.hotel.get_num_rooms_per_category(RoomCategory(c + 1)))
                    for c in range(self.num_room_categories)
                ]

            info["eval_artifacts"] = payload

        # Move to the next day
        self.today += timedelta(days=1)
        terminated = self.today > self.end_simulation
        observation = self.get_observation()

        return observation, reward, terminated, False, info

    # ------------- RESET Method ------------------
    # Resets the environment to its initial state at the beginning of an episode
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        logger.info("resetting env")
        if seed is not None:
            self.seed = seed
            self._init_dynamics_rngs(seed)
            self.acceptance_model.rng = self.acceptance_rng
            self.py_rng = random.Random(seed)

        super().reset(seed=seed)  # Requires this call to control randomness and reproducibility

        # Regenerate requests (if it's required)
        if options is None:
            options = {}
        regen = bool(options.get("new_requests", False))
        if regen:
            requests_seed = options.get("requests_seed", seed)
            self.request_generator.regenerate(seed=requests_seed)

        # Reset the state of the environment to the initial state
        self.today = self.start_simulation
        self.occupancy.fill(0)
        self.demand.fill(0)
        self.bar_levels.fill(1)
        self.accepted_by_day.fill(0)
        self.rejected_by_day.fill(0)
        self.accepted_total = 0
        self.rejected_total = 0

        self.last_total_requests = np.array([], dtype=object)
        self.last_valid_requests = np.array([], dtype=object)
        self.last_valid_prices = np.array([], dtype=float)
        self.last_final_requests = np.array([], dtype=object)
        self.last_final_prices = np.array([], dtype=float)
        self.last_day_avg_price = 0.0

        # Return the initial observation and info
        return self.get_observation(), {}

    # --------- Hotel Availability and Occupancy -----------
    # Method to check hotel availability for a given request
    def check_hotel_availability(self, request: Request | HotelOffer) -> bool:
        if not self._within_booking_window(request):
            return False
        # Get the category and period
        category = request.category
        start_day = request.t_reservation
        days_stay = request.days_stay
        end_day = start_day + timedelta(days=days_stay)
        total_rooms_for_category = self.hotel.get_num_rooms_per_category(category)

        # Get the corresppnding indexes
        category_idx = category.value - 1
        start_idx = (start_day - self.start_simulation).days
        end_idx = (end_day - self.start_simulation).days

        # Check if the period is valid
        if start_day < self.start_simulation or end_day > self.start_simulation + timedelta(
            days=self.extended_simulation_days
        ):
            raise NotValidPeriodError(
                start_period=start_day,
                end_period=end_day,
                start_simulation=self.start_simulation,
                end_simulation=self.start_simulation
                + timedelta(days=self.extended_simulation_days),
            )

        # Return True if all days in the requested period have available rooms, False otherwise
        return bool(
            np.all(self.occupancy[category_idx, start_idx:end_idx] < total_rooms_for_category)
        )

    # Method to update hotel occupancy for a given request
    def update_hotel_occupancy(self, request: Request | HotelOffer) -> bool:
        if not self._within_booking_window(request):
            return False
        # Get the category and period
        category = request.category
        start_day = request.t_reservation
        days_stay = request.days_stay
        end_day = start_day + timedelta(days=days_stay)

        # Get the corresponding indexes
        category_idx = category.value - 1
        start_idx = (start_day - self.start_simulation).days
        end_idx = (end_day - self.start_simulation).days

        total_rooms_for_category = self.hotel.get_num_rooms_per_category(category)

        # Raise an error if the end index exceeds the extended simulation days
        if end_idx > self.extended_simulation_days:
            raise IndexError(
                f"The end index {end_idx} exceeds the extended simulation days {self.extended_simulation_days}."
            )

        # Check capacity for each day of the period
        if np.any(self.occupancy[category_idx, start_idx:end_idx] >= total_rooms_for_category):
            return False

        # Update the occupancy matrix for the specified category and period
        self.occupancy[category_idx, start_idx:end_idx] += 1
        return True

    # Method to update hotel demand based on a list of reservation requests
    def update_hotel_demand(self, requests: np.ndarray) -> None:
        for request in requests:
            request = cast("Request", request)
            category_idx = request.category.value - 1
            t_reservation_idx = (request.t_reservation - self.start_simulation).days
            # Check if the reservation index is within the valid range
            if t_reservation_idx < 0 or t_reservation_idx >= self.extended_simulation_days:
                raise IndexError(
                    f"The reservation index {t_reservation_idx} is out of bounds for the extended simulation days {self.extended_simulation_days}."
                )
            self.demand[category_idx, t_reservation_idx] += 1

    # --------------- Process Requests ---------------
    # Process all incoming requests and propone alternative offers if necessary
    def _process_requests(self, requests: np.ndarray, today: date) -> tuple[np.ndarray, np.ndarray]:
        # Initialize masks and offers lists for tracking request outcomes
        masks: list[bool] = []
        offers: list[HotelOffer] = []
        offer: HotelOffer | None
        # Process each request to determine if it can be accommodated and track the outcomes
        for request in requests:
            # Create an initial hotel offer based on the request's preferred category and dates
            offer = HotelOffer(request, request.category, request.t_reservation, request.days_stay)
            if not self._within_booking_window(offer):
                masks.append(False)
                continue
            # Check if the initial offer can be accommodated in the hotel
            if not self.check_hotel_availability(offer):
                offer = None
                if self.allow_category_alternative:
                    offer = self._check_alternative_room(request)
                if offer is None and self.allow_period_alternative:
                    offer = self._check_alternative_period(request, today)
            masks.append(offer is not None)
            # Append the offer to the offers list if it is not None
            if offer is not None:
                offers.append(offer)
        # Convert the masks and offers lists to numpy arrays for efficient processing and return them
        return np.asarray(masks, dtype=bool), np.asarray(offers, dtype=object)

    # Method to check for an alternative room for a given request
    def _check_alternative_room(self, request: Request) -> HotelOffer | None:
        for category in ROOM_ALTERNATIVES.get(request.category, []):
            offer = HotelOffer(request, category, request.t_reservation, request.days_stay)
            if self.check_hotel_availability(offer):
                return offer
        return None

    # Method to check for an alternative period for a given request
    def _check_alternative_period(self, request: Request, today: date) -> HotelOffer | None:
        alternatives: list[HotelOffer] = []

        # Alternative 0: x+1, d-1 (start one day later)
        if request.days_stay > 1:
            alternatives.append(
                HotelOffer(
                    request,
                    request.category,
                    request.t_reservation + timedelta(days=1),
                    request.days_stay - 1,
                )
            )

        # Alternative 1: x, d-1 (end one day earlier)
        if request.days_stay > 1:
            alternatives.append(
                HotelOffer(request, request.category, request.t_reservation, request.days_stay - 1)
            )

        # Alternative 2: x-1, d (shift one day earlier)
        if request.t_reservation > today + timedelta(days=1):
            alternatives.append(
                HotelOffer(
                    request,
                    request.category,
                    request.t_reservation - timedelta(days=1),
                    request.days_stay,
                )
            )

        # Alternative 3: x+1, d (shift one day later)
        if request.t_reservation + timedelta(days=request.days_stay) < self.extended_end_simulation:
            alternatives.append(
                HotelOffer(
                    request,
                    request.category,
                    request.t_reservation + timedelta(days=1),
                    request.days_stay,
                )
            )

        valid_alternatives = [
            alternative
            for alternative in alternatives
            if self.check_hotel_availability(alternative)
        ]

        if valid_alternatives:
            proposal = valid_alternatives[
                int(self.alternative_rng.integers(len(valid_alternatives)))
            ]
            return proposal

        return None

    # --------------- Apply Mask to Requests ---------------
    # Apply a boolean mask to filter requests based on their validity
    def _apply_mask(self, mask: np.ndarray, requests: np.ndarray) -> np.ndarray:
        return requests[mask.astype(bool)]

    # --------------- Update Bar Levels ---------------
    def _validate_action(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=float)
        if actions.shape != self.action_space.shape:
            raise ValueError(
                f"Action must have shape {self.action_space.shape}, got {actions.shape}"
            )
        if not np.all(np.isfinite(actions)) or np.any((actions < 0) | (actions > 1)):
            raise ValueError("Action values must be finite and in [0, 1]")
        return actions

    def _bar_increment_probabilities(self, actions: np.ndarray) -> np.ndarray:
        actions = self._validate_action(actions)
        tomorrow_idx, stop_idx = self._booking_window_indices()
        occupancy = self.occupancy[:, tomorrow_idx:stop_idx]
        seasons = self.seasonality_array[tomorrow_idx:stop_idx]
        probabilities = np.zeros(occupancy.shape, dtype=float)
        for cat_idx, category in enumerate(RoomCategory):
            rooms = self.hotel.get_num_rooms_per_category(category)
            if rooms < 0:
                raise ValueError("Room capacity must be non-negative")
            if rooms == 0:
                if np.any(occupancy[cat_idx] != 0):
                    raise ValueError("Occupancy must be zero for a category without rooms")
                continue
            for season in range(2):
                mask = seasons == season
                model = BernsteinProbModel(actions[cat_idx, season])
                probabilities[cat_idx, mask] = model.predict_increment_probs(
                    occupancy[cat_idx, mask] / rooms
                )
        return probabilities

    def _update_bar_for_requests_vectorized(self, actions: np.ndarray) -> None:
        probabilities = self._bar_increment_probabilities(actions)
        tomorrow_idx, stop_idx = self._booking_window_indices()
        update_mask = self.np_rng.random(probabilities.shape) < probabilities
        self.bar_levels[:, tomorrow_idx:stop_idx][update_mask] = (
            self.bar_levels[:, tomorrow_idx:stop_idx][update_mask] + 1
        ).clip(max=self._max_bar_level)

    def _update_bar_for_requests(self, actions: np.ndarray) -> None:
        probabilities = self._bar_increment_probabilities(actions)
        tomorrow_idx = (self.today - self.start_simulation).days + 1
        for cat_idx in range(self.num_room_categories):
            for offset, probability in enumerate(probabilities[cat_idx]):
                day_idx = tomorrow_idx + offset
                if self.np_rng.random() < probability:
                    self.bar_levels[cat_idx, day_idx] = min(
                        self.bar_levels[cat_idx, day_idx] + 1, self._max_bar_level
                    )

    # -------- Prices Generation --------
    def _generate_prices(
        self, valid_requests: np.ndarray, actions: np.ndarray
    ) -> list[list[float]]:
        # Update the bar levels based on the actions before generating prices
        self._update_bar_for_requests_vectorized(actions)
        return self._quote_prices(valid_requests)

    def _quote_prices(self, valid_requests: np.ndarray) -> list[list[float]]:
        # Initialize the list to store prices for all requests
        prices_all: list[list[float]] = []

        # Loop over each valid request to generate prices
        for req in valid_requests:
            req = cast("HotelOffer", req)

            cat_idx, start_idx, end_idx = self._stay_indices(req)

            daily_prices: list[float] = []
            for day_idx in range(start_idx, end_idx):
                # Take the bar level, that is updated before
                bar_level = int(self.bar_levels[cat_idx, day_idx])
                # Get the price from the bar table
                price = get_price(req.category.name, bar_level)
                daily_prices.append(price)
            prices_all.append(daily_prices)

        return prices_all

    def _evaluate_prices(
        self, daily_prices: list[list[float]], requests: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(daily_prices) != len(requests):
            raise ValueError(
                f"The number of daily prices must be equal to the number of requests. Daily prices: {len(daily_prices)}, Requests: {len(requests)}."
            )
        accepted_prices = []
        total_prices = []
        for prices, request in zip(daily_prices, requests, strict=True):
            total_price = sum(prices)
            flag = self.acceptance_model.accept(total_price, request)

            accepted_prices.append(flag)
            total_prices.append(total_price)
        return np.array(accepted_prices, dtype=bool), np.array(total_prices, dtype=float)

    def _confirm_offers(
        self,
        offers: np.ndarray,
        daily_prices: list[list[float]],
        accepted_mask: np.ndarray,
        total_prices: np.ndarray,
    ) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
        confirmed = np.zeros(len(offers), dtype=bool)
        attempts: list[list[dict[str, Any]]] = [[] for _ in offers]
        for index, offer in enumerate(offers):
            accepted = bool(accepted_mask[index])
            booked = accepted and self.update_hotel_occupancy(offer)
            if self.collect_eval_artifacts:
                attempts[index].append(
                    self._offer_attempt_to_artifact(offer, daily_prices[index], accepted, booked)
                )
            if booked:
                confirmed[index] = True
                continue
            if not accepted:
                continue
            _, alternatives = self._process_requests(
                np.asarray([offer.request], dtype=object), self.today
            )
            if not len(alternatives):
                continue
            alternative = alternatives[0]
            prices = self._quote_prices(alternatives)[0]
            total = float(sum(prices))
            accepted = bool(self.acceptance_model.accept(total, alternative))
            booked = accepted and self.update_hotel_occupancy(alternative)
            offers[index] = alternative
            daily_prices[index] = prices
            total_prices[index] = total
            accepted_mask[index] = accepted
            confirmed[index] = booked
            if self.collect_eval_artifacts:
                attempts[index].append(
                    self._offer_attempt_to_artifact(alternative, prices, accepted, booked)
                )
        return confirmed, attempts

    @staticmethod
    def _offer_attempt_to_artifact(
        offer: HotelOffer, prices: list[float], accepted: bool, confirmed: bool
    ) -> dict[str, Any]:
        return {
            "category": int(offer.category.value),
            "check_in": offer.t_reservation.isoformat(),
            "days_stay": int(offer.days_stay),
            "period_changed": offer.period_changed,
            "daily_prices": [float(price) for price in prices],
            "price_accepted": accepted,
            "confirmed": confirmed,
        }

    # --------- Rewards and Updates -----------
    def _calculate_reward(self, prices: np.ndarray) -> float:
        return float(np.sum(prices))

    # Update the occupancy of the hotel based on the accepted requests
    def _update_occupancy(self, accepted_requests: np.ndarray) -> np.ndarray:
        # Initialize a list to store the validity of each accepted request
        valid_mask = []
        # Iterate over each accepted request
        for request in accepted_requests:
            # Check if the request can be accommodated and update occupancy accordingly
            valid_mask.append(self.update_hotel_occupancy(request))
        return np.array(valid_mask, dtype=bool)

    def _update_demand(self, requests: np.ndarray) -> None:
        self.update_hotel_demand(requests)

    # --------------- Helper Methods for Plots -------------
    def get_num_rooms_per_category(self, category: RoomCategory) -> int:
        return self.hotel.get_num_rooms_per_category(category)

    # To serialize a request without having a mutable reference
    @staticmethod
    def _request_to_artifact(request: Request) -> dict[str, Any]:
        return {
            "request_date": request.t_request.isoformat(),
            "check_in": request.t_reservation.isoformat(),
            "days_stay": int(request.days_stay),
            "category": int(request.category.value),
            "num_guests": int(request.num_guests),
        }


if __name__ == "__main__":
    import time
    from datetime import date

    import numpy as np
    from loguru import logger

    hotel = Hotel(num_category1_rooms=10, num_category2_rooms=20, num_category3_rooms=30)

    # Sanity check on invalid dates
    try:
        _ = PricingEnv(
            start_simulation=date(2025, 12, 31),
            end_simulation=date(2025, 1, 1),
            hotel=hotel,
        )
    except ValueError as e:
        logger.info("ValueError correctly raised for invalid dates.")
        logger.info(e)

    # Valid environment
    env = PricingEnv(
        start_simulation=date(2025, 1, 1),
        end_simulation=date(2025, 1, 31),
        hotel=hotel,
        seed=42,
    )

    obs, info = env.reset()
    logger.info(f"Initial observation OK: keys={list(obs.keys())}")

    s = time.time()
    total_reward = 0.0

    for step_i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        logger.info(f"Step {step_i}: reward={reward:.2f}")

        if terminated or truncated:
            break

    logger.info(f"Simulation completed. Total reward={total_reward:.2f}")
    logger.info(f"Avg step time: {(time.time() - s) / (step_i + 1):.4f} sec/step")
