import random
from copy import deepcopy
from datetime import date, timedelta
from typing import ClassVar, cast

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
from pricing_advertising.pricing_model.bernstein_model import BernsteinPolynomialModel
from pricing_advertising.requests.request import Request
from pricing_advertising.requests.request_generator import RequestGenerator
from pricing_advertising.utils.constants import (
    LOGISTIC_ACCEPTANCE_AVG_PRICES,
    LOGISTIC_ACCEPTANCE_BETA,
    LOGISTIC_ACCEPTANCE_VARIANCES,
    MAX_AVG_NUM_REQUESTS,
    MAX_PRICE,
    MIN_AVG_NUM_REQUESTS,
    MIN_PRICE,
    NUM_CATEGORIES,
    NUM_COEFFICIENTS,
)
from pricing_advertising.utils.utils_functions import extract_advance


class PricingEnv(gym.Env):
    # Metadata is a required attribute for the environment
    metadata: ClassVar[dict] = {"render.modes": ["human"], "render_fps": 1}

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

        self.seed = seed
        self.np_rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)

        # Define the dimensions of the state and action spaces and other vars/structure data you need
        self.num_room_categories = len(RoomCategory)
        self.total_days = (end_simulation - start_simulation).days

        # Demand and Occupancy of the hotel
        self.occupancy = np.zeros((NUM_CATEGORIES, self.total_days), dtype=int)
        self.demand = np.zeros((NUM_CATEGORIES, self.total_days), dtype=int)

        # Define the action and observation space
        # Obsevation space includes hotel occupancy, past demand, seasonality
        self.observation_space = gym.spaces.Dict(
            {
                "occupancy": gym.spaces.Box(
                    low=0, high=1, shape=(self.num_room_categories, self.total_days)
                ),
                # 1000 da parametrizzare: max customer
                "demand": gym.spaces.Box(
                    low=0, high=1, shape=(self.num_room_categories, self.total_days)
                ),
                # represent a day as a integer number
                # TODO: coversion sin and cosine
                "today": gym.spaces.Box(low=0, high=1),
                # TODO: remember to add seasonality
                "seasonality": gym.spaces.Discrete(2, seed=seed),
            }
        )

        # action space is coefficients of the Bernstein polynomial
        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.num_room_categories, NUM_COEFFICIENTS),
            dtype=np.float32,
        )

        # Initiliaze the state
        self.today = start_simulation
        self.start_simulation = start_simulation
        self.end_simulation = end_simulation
        self.avg_num_requests = avg_num_requests or random.randint(
            MIN_AVG_NUM_REQUESTS, MAX_AVG_NUM_REQUESTS
        )
        self.hotel = hotel

        # Instantiate the request generator and acceptance model
        self.request_generator = request_generator or RequestGenerator(
            self.avg_num_requests, end_simulation, distribution=None, seed=seed
        )
        self.acceptance_model = acceptance_model or LogisticAcceptanceModel(
            beta=LOGISTIC_ACCEPTANCE_BETA,
            avg_prices=LOGISTIC_ACCEPTANCE_AVG_PRICES,
            variances=LOGISTIC_ACCEPTANCE_VARIANCES,
        )

    # region STEP
    def step(self, action) -> tuple:
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
        requests = self.request_generator.generate_requests(self.today)

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

        # Calculate the revenue of accepted requests
        reward = self._calculate_reward(final_prices, final_requests)  # to fix

        # Move to the next day
        self.today += timedelta(days=1)
        done = self.today >= self.end_simulation
        observation = self.observation_space.sample()
        info = {}  # Placeholder for additional info

        return observation, reward, done, info

    # endregion

    def reset(self, seed: int | None = None) -> tuple:
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

        # Reset the state of the environment to the initial state
        self.today = self.start_simulation

        # Return the initial observation and info
        return self.observation_space.sample(), {}

    # OTHER FUNCTIONS
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
        if start_day < self.start_simulation or end_day > self.end_simulation:
            raise NotValidPeriodError(
                start_period=start_day,
                end_period=end_day,
                start_simulation=self.start_simulation,
                end_simulation=self.end_simulation,
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
        category_idx = category.value
        start_idx = (start_day - self.start_simulation).days
        end_idx = (end_day - self.start_simulation).days

        total_rooms_for_category = self.hotel.get_num_rooms_per_category(category)

        # Check capacity for each day of the period
        if np.any(self.occupancy[category_idx - 1, start_idx:end_idx] > total_rooms_for_category):
            return False

        # Update the occupancy matrix for the specified category and period
        self.occupancy[category_idx - 1, start_idx:end_idx] += 1
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
            request = cast(Request, request)
            category_idx = request.category.value - 1
            t_reservation_idx = (request.t_reservation - self.start_simulation).days
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
            if not self.check_hotel_availability(request):
                alternative_room = self._check_alternatives(request)
                if alternative_room is None:
                    masks.append(0)
                else:
                    masks.append(1)
            else:
                masks.append(1)
        return np.array(masks)

    def _check_alternatives(self, request: Request) -> RoomCategory | None:
        """
        Checks for alternative room categories if the requested category is unavailable.

        Args:
            request (Request): The request object containing the desired room category.

        Returns:
            RoomCategory | None: An alternative room category if available, or None if no alternatives are found
            or if the requested category is CATEGORY3 (which has no alternatives).
        """
        if request.category is RoomCategory.CATEGORY3:
            return None
        original_category = request.category
        alternatives_category = ROOM_ALTERNATIVES.get(original_category, [])
        for alt_category in alternatives_category:
            modified_request = deepcopy(request)
            modified_request.category = alt_category
            if self.check_hotel_availability(modified_request):
                return alt_category

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

    def _generate_prices(
        self, valid_requests: np.ndarray, actions: np.ndarray
    ) -> list[list[float]]:
        """
        Generate a list of price predictions for a set of valid requests.

        This method calculates the predicted prices for each request based on
        the advance days, reservation details, and a pricing model defined by
        Bernstein polynomial coefficients.

        Args:
            valid_requests (np.ndarray): An array of valid requests, where each
                request contains details such as reservation time, category, and
                total reservation days.

        Returns:
            list[list[float]]: A nested list where each inner list contains the
                predicted prices for each day of the reservation for a specific
                request.
        """
        requests_prices = []
        advances = [extract_advance(req) for req in valid_requests]
        advances = np.array(advances)
        pricing_model = BernsteinPolynomialModel(actions)  # action are theta coefficients

        for advance, request in zip(advances, valid_requests, strict=True):
            request_prices = []
            request = cast(Request, request)

            for day_idx in range(request.total_reservation_days):
                price = pricing_model.predict_price((advance - day_idx) / self.total_days)
                request_prices.append(price)
            requests_prices.append(request_prices)
        return requests_prices

    def _generate_theta_coefficients(self, category: RoomCategory, season: str) -> np.ndarray:
        """
        Generate theta coefficients for a given room category and season.

        This method generates an array of random coefficients (theta) based on the
        specified room category and season. The coefficients are sampled uniformly
        from predefined ranges depending on the category and season.

        Args:
            category (RoomCategory): The category of the room. Must be one of
                RoomCategory.CATEGORY1, RoomCategory.CATEGORY2, or RoomCategory.CATEGORY3.
            season (str): The season for which the coefficients are generated.
                Must be either "low" or "high".

        Returns:
            np.ndarray: An array of randomly generated theta coefficients.

        Raises:
            InvalidRoomCategoryError: If the provided room category is invalid.
        """
        num_coefficients = NUM_COEFFICIENTS
        rng = np.random.default_rng()

        if season == "low":
            if category == RoomCategory.CATEGORY1:
                theta = rng.uniform(0.5, 1.5, size=num_coefficients)
            elif category == RoomCategory.CATEGORY2:
                theta = rng.uniform(1.0, 3.0, size=num_coefficients)
            elif category == RoomCategory.CATEGORY3:
                theta = rng.uniform(2.0, 5.0, size=num_coefficients)
            else:
                raise InvalidRoomCategoryError(
                    "The room catgory {category} is invalid. Choose one of the available categories."
                )
        elif season == "high":
            if category == RoomCategory.CATEGORY1:
                theta = rng.uniform(1.5, 3.0, size=num_coefficients)
            elif category == RoomCategory.CATEGORY2:
                theta = rng.uniform(3.0, 6.0, size=num_coefficients)
            elif category == RoomCategory.CATEGORY3:
                theta = rng.uniform(5.0, 9.0, size=num_coefficients)
            else:
                raise InvalidRoomCategoryError(
                    "The room catgory {category} is invalid. Choose one of the available categories."
                )
        return theta

    def _generate_random_prices_for_request(self, request: Request) -> list[float]:
        """
        Generate a list of random prices for each day of a reservation request.

        This method generates random daily prices within the range defined by
        MIN_PRICE and MAX_PRICE for the total number of reservation days specified
        in the request.

        Args:
            request (Request): The reservation request object containing details
                               about the reservation.

        Returns:
            list[float]: A list of random daily prices for the reservation period.
        """
        request_prices = []
        reservation_days = request.total_reservation_days
        for _ in range(reservation_days):
            rng = np.random.default_rng()
            random_daily_price = rng.uniform(MIN_PRICE, MAX_PRICE)
            request_prices.append(random_daily_price)
        return request_prices

    def _generate_random_prices(self, valid_requests: np.ndarray) -> list[list[float]]:
        """
        Generate random prices for a list of valid requests.

        Args:
            valid_requests (np.ndarray): An array of valid requests for which prices need to be generated.

        Returns:
            list[list[float]]: A list of lists, where each inner list contains the generated prices for a corresponding request.
        """
        requests_prices = []
        for request in valid_requests:
            request_prices = self._generate_random_prices_for_request(request)
            requests_prices.append(request_prices)
        return requests_prices

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


# region MAIN
if __name__ == "__main__":
    hotel = Hotel(num_category1_rooms=10, num_category2_rooms=20, num_category3_rooms=30)

    try:
        env_invalid_dates = PricingEnv(
            start_simulation=date(2025, 12, 31),
            end_simulation=date(2025, 1, 1),
            hotel=hotel,
        )
    except ValueError as e:
        logger.info("Value error succesfully triggered.")
        logger.info(e)

    env = PricingEnv(
        start_simulation=date(2025, 1, 1),
        end_simulation=date(2025, 1, 31),
        hotel=hotel,
    )

    obs, info = env.reset()
    logger.info(f"Initial observation: {obs}")
    import time

    s = time.time()
    for _ in range(100):
        action = env.action_space.sample()
        logger.info(f"Action: {action}")
        step_start = time.time()
        obs, reward, done, info = env.step(action)
        # logger.info(f"Observation: {obs}, Reward: {reward}, Done: {done}, Info: {info}")
        step_end = time.time()
        logger.info(f"Step time: {step_end - step_start} seconds")
        if done:
            break
        e_ = time.time()
        logger.info(f"Time taken: {(e_ - s)} seconds")
    logger.info("Simulation completed.")

    e = time.time()
    logger.info(f"Time taken: {(e - s) / 100} seconds")
# endregion
