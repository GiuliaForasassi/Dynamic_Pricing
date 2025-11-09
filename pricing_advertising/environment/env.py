import random
from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
from typing import Any, cast

import gymnasium as gym
import numpy as np
from loguru import logger

from pricing_advertising.acceptance.logistic_function import LogisticAcceptanceModel
from pricing_advertising.acceptance.simple_function import ABCAcceptanceModel
from pricing_advertising.exceptions.exceptions import (
    InvalidRoomCategoryError,
    NotValidPeriodError,
)
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.hotel.room import ROOM_ALTERNATIVES, RoomCategory
from pricing_advertising.pricing_model.bar_table import BAR_TABLE, n_levels
from pricing_advertising.pricing_model.bernstein_model import BernsteinProbModel
from pricing_advertising.requests.request import Request
from pricing_advertising.requests.request_generator import RequestGenerator
from pricing_advertising.utils.constants import (
    HIGH_SEASON,
    LOGISTIC_ACCEPTANCE_AVG_PRICES,
    LOGISTIC_ACCEPTANCE_BETA,
    LOGISTIC_ACCEPTANCE_VARIANCES,
    MAX_AVG_NUM_REQUESTS,
    MIN_AVG_NUM_REQUESTS,
    NUM_COEFFICIENTS,
    NUM_WINDOWS,
    WINDOW_SIZE,
)


class PricingEnv(gym.Env):
    # Metadata is a required attribute for the environment
    # metadata: ClassVar[dict] = {"render.modes": ["human"], "render_fps": 1}  # type: ignore

    def __init__(
        self,
        start_simulation: date,
        end_simulation: date,
        hotel: Hotel,
        avg_num_requests: int | None = None,
        request_generator: RequestGenerator | None = None,
        acceptance_model: ABCAcceptanceModel | None = None,
        seed: int | None = None,
    ) -> None:
        if start_simulation > end_simulation:
            raise ValueError(
                f"The start date {start_simulation} must be before the end date {end_simulation}."
            )

        super().__init__()

        # ============ RANDOM AND SEED SETUP ============
        self.seed = seed
        self.np_rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)

        # ============ SIMULATION HORIZON AND HOTEL ============
        self.start_simulation = start_simulation
        self.end_simulation = end_simulation
        self.window_span_days = NUM_WINDOWS * WINDOW_SIZE
        self.extended_end_simulation = end_simulation + timedelta(days=self.window_span_days)

        self.total_days = (end_simulation - start_simulation).days
        self.extended_simulation_days = self.total_days + self.window_span_days

        self.hotel = hotel
        self.num_room_categories = len(RoomCategory)

        # ============ STATE VARIABLES ============
        self.today = start_simulation
        self.n_levels = n_levels()
        self._max_bar_level = self.n_levels - 1

        self.occupancy = np.zeros(
            (self.num_room_categories, self.extended_simulation_days), dtype=int
        )
        self.demand = np.zeros((self.num_room_categories, self.extended_simulation_days), dtype=int)
        self.bar_levels = np.zeros(
            (self.num_room_categories, self.extended_simulation_days), dtype=int
        )

        # Define the action and observation space in Gym

        # =========== OBSERVATION SPACE ============
        self.observation_space = gym.spaces.Dict(
            {
                "occupancy": gym.spaces.Box(
                    low=0, high=1, shape=(self.num_room_categories, NUM_WINDOWS), dtype=np.float32
                ),
                # 1000 da parametrizzare: max customer
                "demand": gym.spaces.Box(
                    low=0, high=1, shape=(self.num_room_categories, NUM_WINDOWS), dtype=np.float32
                ),
                # represent a day as a integer number
                "today": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                # seasonality is represented as a discrete value: 0 for low season, 1 for high season
                "seasonality": gym.spaces.Discrete(2, seed=seed),
                # DYNAMIC: FOR EACH WINDOW
                # profile of BARs by category and by window --> represents where we are (current status)
                "bar_levels": gym.spaces.Box(
                    low=0, high=1, shape=(self.num_room_categories, NUM_WINDOWS), dtype=np.float32
                ),
            }
        )

        # =========== ACTION SPACE ============
        # Action space is coefficients of the Bernstein polynomial
        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.num_room_categories, NUM_COEFFICIENTS),
            dtype=np.float32,
        )

        # =========== REQUEST GENERATOR AND ACCEPTANCE MODEL ============
        self.avg_num_requests = avg_num_requests or random.randint(
            MIN_AVG_NUM_REQUESTS, MAX_AVG_NUM_REQUESTS
        )

        self.seasonality_array = np.array(
            [
                self.get_seasonality(start_simulation + timedelta(day_idx))
                for day_idx in range((self.extended_end_simulation - self.start_simulation).days)
            ]
        )

        # Instantiate the request generator and acceptance model
        self.request_generator = request_generator or RequestGenerator(
            self.avg_num_requests,
            start_simulation,
            end_simulation,
            extended_end_simulation=self.extended_end_simulation,
            seasonality_array=self.seasonality_array,
            distribution=None,
            seed=seed,
            precompute=True,  # always precompute requests at the beginning
        )

        if request_generator is not None:
            self.request_generator.ensure_precomputed()

        self.acceptance_model = acceptance_model or LogisticAcceptanceModel(
            beta=LOGISTIC_ACCEPTANCE_BETA,
            avg_prices=LOGISTIC_ACCEPTANCE_AVG_PRICES,
            variances=LOGISTIC_ACCEPTANCE_VARIANCES,
        )

        # Attribute for counting accepted/rejected requests
        self.accepted_by_day = np.zeros(
            (self.num_room_categories, self.extended_simulation_days), dtype=np.int32
        )
        self.rejected_by_day = np.zeros_like(self.accepted_by_day)

        self.accepted_total = 0
        self.rejected_total = 0

    # ========== GET OBSERVATION ==========
    def get_observation(self) -> dict:
        current_day_index = (self.today - self.start_simulation).days
        simulation_days = (self.end_simulation - self.start_simulation).days

        occupancy_windows = np.zeros((self.num_room_categories, NUM_WINDOWS), dtype=np.float32)
        demand_windows = np.zeros((self.num_room_categories, NUM_WINDOWS), dtype=np.float32)
        bar_levels_windows = np.zeros((self.num_room_categories, NUM_WINDOWS), dtype=np.float32)

        # The environment aims to create an “aggregate vision” of the near future for each room category
        for category in range(self.num_room_categories):
            num_rooms = self.hotel.get_num_rooms_per_category(RoomCategory(category + 1))

            for window in range(NUM_WINDOWS):
                # Calculate the start and end indices for the current window
                start = current_day_index + window * WINDOW_SIZE
                end = min(start + WINDOW_SIZE, self.extended_simulation_days)
                width = max(1, end - start)
                if width == 0:
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

        today_norm = np.array(
            [min(1.0, max(0.0, current_day_index / max(1, simulation_days)))],
            dtype=np.float32,
        )

        seasonality = self.seasonality_array

        # Clip for security
        occupancy_windows = np.clip(occupancy_windows, 0.0, 1.0)
        demand_windows = np.clip(demand_windows, 0.0, 1.0)
        bar_levels_windows = np.clip(bar_levels_windows, 0.0, 1.0)

        return {
            "occupancy": occupancy_windows,
            "demand": demand_windows,
            "today": today_norm,
            "seasonality": seasonality,
            "bar_levels": bar_levels_windows,  # dynamic
        }

    def get_seasonality(self, time: date) -> int:
        """
        Determine the seasonality based on the given date.

        Args:
            time (date): The date for which to determine the seasonality.

        Returns:
            int: Returns 1 if the month of the given date is considered high season
                 (January, February, July, August, September, or December),
                 otherwise returns 0.
        """
        return int(time.month in HIGH_SEASON)

    def _stay_indices(self, req: Request) -> tuple[int, int, int]:
        cat_idx = req.category.value - 1
        start_idx = (req.t_reservation - self.start_simulation).days
        end_idx = start_idx + req.days_stay

        if cat_idx < 0 or cat_idx >= self.num_room_categories:
            raise InvalidRoomCategoryError(req.category.value)
        if start_idx < 0 or end_idx > self.extended_simulation_days:
            raise NotValidPeriodError(
                start_period=req.t_reservation,
                end_period=req.t_reservation + timedelta(days=req.days_stay),
                start_simulation=self.start_simulation,
                end_simulation=self.start_simulation
                + timedelta(days=self.extended_simulation_days),
            )
        return cat_idx, start_idx, end_idx

    # ========== STEP METHOD ==========
    def step(self, action) -> tuple:  # noqa: ANN001
        """
        Execute one time step within the environment.

        Parameters:
        action (object): An action provided by the environment.

        Returns:
        tuple: A tuple containing:
            - observation (object): An observation of the environment.
            - reward (float): The amount of reward returned after the previous action.
            - done (bool): Whether the episode has ended.
            - info (dict): Additional information about the environment.
        """

        # Generate requests for the current day
        requests = self.request_generator.fetch_requests_for_date(self.today)

        # Check if the requests can be accommodated
        valid_mask = self._process_requests(requests)
        valid_requests = self._apply_mask(valid_mask, requests)

        # Update the demand
        self._update_demand(requests)

        # Generate prices for each day of valid requests
        daily_prices = self._generate_prices(valid_requests, action)

        # Accept or reject the requests
        (accepted_mask, tot_prices) = self._evaluate_prices(daily_prices, valid_requests)
        accepted_requests = self._apply_mask(accepted_mask, valid_requests)
        accepted_prices = self._apply_mask(accepted_mask, tot_prices)

        # Update the hotel occupancy
        occupancy_mask = self._update_occupancy(accepted_requests)
        final_requests = self._apply_mask(occupancy_mask, accepted_requests)
        final_prices = self._apply_mask(occupancy_mask, accepted_prices)

        # =========================To calculate accepted requests======================

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

        # (facoltativo) debug print ogni 30 giorni
        if (current_day_index % 30) == 0:
            print(
                f"[ACCEPT-COUNT] day={current_day_index} acc_today={sum(acc_per_cat.values())} "
                f"rej_today={sum(tot_per_cat.values()) - sum(acc_per_cat.values())} "
                f"acc_total={self.accepted_total} rej_total={self.rejected_total}"
            )
        # =============================================================================

        # Calculate the revenue of accepted requests
        reward = self._calculate_reward(final_prices, final_requests)  # to fix

        # Move to the next day
        self.today += timedelta(days=1)
        terminated = self.today >= self.end_simulation
        observation = self.get_observation()
        info: dict = {}  # Placeholder for additional info

        return observation, reward, terminated, False, info

    # ========== RESET METHOD ==========
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,  # noqa: ARG002
    ) -> tuple[Any, dict[str, Any]]:
        """
        Resets the environment to its initial state.

        This method is called to reset the environment to its initial state at the beginning of an episode.
        It also ensures that the randomness and reproducibility are controlled by calling the parent class's reset method.

        Args:
            seed (int, optional): An optional seed for the random number generator to ensure reproducibility. Defaults to None.

        Returns:
            tuple: A tuple containing the initial observation from the observation space and an empty dictionary for additional info.
        """
        if seed is not None:
            self.np_rng = np.random.default_rng(seed)
            self.py_rng = random.Random(seed)

        super().reset(seed=seed)  # requires this call to control randomness and reproducibility

        # Regenerate requests (if it's required)
        if options is None:
            options = {}
        regen = bool(options.get("new_requests", False))
        if regen:
            requests_seed = options.get("requests_seed", seed if seed is not None else self.seed)
            # If RG supports regenerate() use it
            if hasattr(self.request_generator, "regenerate"):
                self.request_generator.regenerate(seed=requests_seed)
            else:
                self.request_generator.requests_by_day.clear()
                self.request_generator._pre_generate_all_requests()  # noqa: SLF001

        # Reset the state of the environment to the initial state
        self.today = self.start_simulation
        self.occupancy.fill(0)
        self.demand.fill(0)
        self.bar_levels.fill(0)
        self.accepted_by_day.fill(0)
        self.rejected_by_day.fill(0)
        self.accepted_total = 0
        self.rejected_total = 0

        # Return the initial observation and info
        return self.get_observation(), {}

    # ========== OTHER METHODS ==========
    def check_hotel_availability(self, request: Request) -> bool:
        """
        Checks the availability of rooms for a given request.

        Args:
            request (Request): The request object containing the reservation details.

        Returns:
            bool: True if there is availability for the requested category and period, False otherwise.

        Raises:
            NotValidPeriodError: If the requested period is not valid.
        """
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

        return bool(
            np.all(self.occupancy[category_idx, start_idx:end_idx] < total_rooms_for_category)
        )

    def update_hotel_occupancy(self, request: Request) -> bool:
        """
        Updates the hotel occupancy for a given reservation request.

        This method checks if a reservation request can be accommodated based on the
        current occupancy and the total number of rooms available for the requested
        category. If the request is valid and there is sufficient capacity, the
        occupancy matrix is updated accordingly.

        Args:
            request (Request): The reservation request containing the category,
                start date, and duration of the stay.

        Returns:
            bool: True if the reservation can be accommodated and the occupancy
                matrix is updated, False otherwise.

        """
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

    def update_hotel_demand(self, requests: np.ndarray) -> None:
        """
        Updates the hotel demand matrix based on a list of reservation requests.

        This method processes each request in the provided list and increments the
        corresponding entry in the demand matrix. The demand matrix is updated
        based on the category of the request and the reservation date relative
        to the simulation start date.

        Args:
            requests (np.ndarray): A vector of Request objects, where each
                request contains information about the category and reservation
                date.

        Returns:
            None
        """
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

    def _process_requests(self, requests: np.ndarray) -> np.ndarray:
        """
        Processes a batch of requests and returns a mask indicating the availability of rooms.

        Args:
            requests (np.ndarray): An array of requests where each request is an object that can be checked for availability.

        Returns:
            np.ndarray: An array of integers (0 or 1) where 1 indicates that the requested room or an alternative room is available, and 0 indicates that neither the requested room nor an alternative room is available.
        """
        masks = []
        for request in requests:
            if self.check_hotel_availability(request):
                masks.append(1)
                continue

            alternative_room = self._check_alternative_room(request)
            if alternative_room is not None:
                masks.append(1)
                continue

            alternative_period = self._check_alternative_period(request)
            if alternative_period is not None:
                request.proposed_t_reservation = alternative_period.proposed_t_reservation
                request.proposed_days_stay = alternative_period.proposed_days_stay
                masks.append(1)
                continue

            masks.append(0)
        return np.array(masks)

    def _check_alternative_room(self, request: Request) -> RoomCategory | None:
        """
        Checks for alternative room categories if the requested category is
        unavailable.

        Args:
            request (Request): The request object containing the desired room
            category.

        Returns:
            RoomCategory | None: An alternative room category if available, or
            None otherwise.
        """
        original_category = request.category
        alternatives_category = ROOM_ALTERNATIVES.get(original_category, [])
        for alt_category in alternatives_category:
            modified_request = deepcopy(request)
            modified_request.category = alt_category
            if self.check_hotel_availability(modified_request):
                return alt_category
        # If no alternatives are available, return None
        return None

    def _check_alternative_period(self, request: Request) -> Request | None:
        """
        Checks for alternative period if the request cannot be satisfied.

        Args:
            request (Request): The request object containing the desired reservation
            category and days.

        Returns:
            Request | None: A request with an alternative period if possible, or
            None otherwise.
        """
        alternatives = [deepcopy(request) for _ in range(4)]

        # alternative 0: x+1, d-1 (start one day later)
        alternatives[0].proposed_t_reservation = request.t_reservation - timedelta(days=1)
        alternatives[0].proposed_days_stay = request.days_stay - 1

        # alternative 1: x, d-1 (end one day earlier)
        alternatives[1].proposed_days_stay = request.days_stay - 1

        # alternative 2: x-1, d (shift one day earlier)
        alternatives[2].proposed_t_reservation = request.t_reservation - timedelta(1)

        # alternative 3: x+1, d (shift one day later)
        alternatives[3].proposed_t_reservation = request.t_reservation + timedelta(1)

        valid_alternatives = [
            alternative
            for alternative in alternatives
            if self.check_hotel_availability(alternative)
        ]

        if valid_alternatives:
            proposal = random.choice(alternatives)
            return proposal

        return None

    def _apply_mask(self, mask: np.ndarray, requests: np.ndarray) -> np.ndarray:
        """
        Apply a mask to filter requests based on a validity mask.

        Parameters:
        mask (np.ndarray): A boolean array indicating which requests consider (0 or 1).
        requests (np.ndarray): An array of requests to be filtered.

        Returns:
        np.ndarray: An array of requests that are filtered according to the choosen mask.
        """
        return requests[mask.astype(bool)]

    # ========== UPDATE BAR LEVELS ==========
    def _update_bar_for_requests(self, actions: np.ndarray) -> None:
        # 1) Create the Bernstein polynomial models for each room category
        models: list[BernsteinProbModel] = []
        for cat_idx in range(self.num_room_categories):
            theta = np.array(actions[cat_idx], dtype=float)
            models.append(BernsteinProbModel(theta))

        # 2) Look at each request
        today_idx = (self.today - self.start_simulation).days
        end_idx = (self.end_simulation - self.start_simulation).days
        # end_idx = self.extended_simulation_days # TODO: like above or like this?
        for cat_idx, category in enumerate(RoomCategory):
            rooms = self.hotel.get_num_rooms_per_category(category)
            for day_idx in range(today_idx + 1, end_idx):
                # 2a) Calculate the occupancy ratio for that day and category
                occ_ratio = float(self.occupancy[cat_idx, day_idx]) / max(1, rooms)
                # If for any reason the occupancy ratio is out of bounds, we clip it between 0 and 1
                occ_ratio = np.clip(occ_ratio, 0.0, 1.0)

                # 2b) Raw probabbility to increase the bar level
                p = models[cat_idx].predict_increment_probs(occ_ratio)

                # 2c) ...

                # 2c) Increase the bar level with probability p --> Bernoulli trial
                if self.np_rng.random() < p:
                    self.bar_levels[cat_idx, day_idx] = min(
                        self.bar_levels[cat_idx, day_idx] + 1, self._max_bar_level
                    )

    # ========== PRICES GENERATION ==========
    def _generate_prices(
        self, valid_requests: np.ndarray, actions: np.ndarray
    ) -> list[list[float]]:
        # 1) Update bar levels
        self._update_bar_for_requests(actions)

        prices_all: list[list[float]] = []

        # 2) Generate prices for each request
        for req in valid_requests:
            req = cast("Request", req)

            cat_idx, start_idx, end_idx = self._stay_indices(req)

            daily_prices: list[float] = []
            for day_idx in range(start_idx, end_idx):
                # Take the bar level, that is updated before (point 1)
                bar_level = int(self.bar_levels[cat_idx, day_idx])
                # Get the price from the bar table
                price = BAR_TABLE[req.category][bar_level]
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
        return np.array(accepted_prices), np.array(total_prices)

    def _calculate_reward(self, prices: np.ndarray, accepted_mask: np.ndarray) -> float:
        """
        Calculate the reward generated by the accepted requests.

        Parameters:
        prices (np.ndarray): An array of prices proposed to the customers.
        accepted_mask (np.ndarray): A boolean array indicating which requests were accepted.

        Returns:
        float: The total reward generated by the accepted requests.
        """
        return np.sum(prices[accepted_mask.astype(bool)])

    def _update_occupancy(self, accepted_requests: np.ndarray) -> np.ndarray:
        """
        Update the occupancy of the hotel based on the accepted requests.

        Parameters:
        accepted_requests (np.ndarray): An array of accepted requests.
        """
        valid_mask = []
        for request in accepted_requests:
            valid_mask.append(self.update_hotel_occupancy(request))
        return np.array(valid_mask, dtype=bool)

    def _update_demand(self, requests: np.ndarray) -> None:
        """
        Update the demand for the current day based on the generated requests.

        Parameters:
        requests (np.ndarray): An array of requests generated for the current day.
        """
        self.update_hotel_demand(requests)

    # ========== HELPER FOR DOING PLOTS ==========
    def get_num_rooms_per_category(self, category: RoomCategory) -> int:
        return self.hotel.get_num_rooms_per_category(category)


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
