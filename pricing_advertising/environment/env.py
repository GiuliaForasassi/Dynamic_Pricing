import random
from datetime import date, timedelta
from typing import ClassVar

import gymnasium as gym
import numpy as np

from pricing_advertising.acceptance.simple_function import SimpleAcceptanceModel
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.hotel.room import RoomCategory
from pricing_advertising.requests.request import Request
from pricing_advertising.requests.request_generator import RequestGenerator
from pricing_advertising.utils.constants import (
    MAX_AVG_NUM_REQUESTS,
    MAX_PRICE,
    MIN_AVG_NUM_REQUESTS,
    MIN_PRICE,
    NUM_COEFFICIENTS,
)


class PricingEnv(gym.Env):
    # Metadata is a required attribute for the environment
    metadata: ClassVar[dict] = {"render.modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        start_simulation: date,
        end_simulation: date,
        hotel: Hotel,
    ) -> None:
        super().__init__()

        # Define the dimensions of the state and action spaces and other vars/structure data you need
        num_room_categories = len(RoomCategory)
        total_days = (end_simulation - start_simulation).days

        # Define the action and observation space
        # Obsevation space includes hotel occupancy, past demand, seasonality
        self.observation_space = gym.spaces.Dict(
            {
                "occupancy": gym.spaces.Box(low=0, high=1, shape=(num_room_categories, total_days)),
                # 1000 da parametrizzare: max customer
                "demand": gym.spaces.Box(low=0, high=1000, shape=(num_room_categories, total_days)),
                # represent a day as a integer number
                # TODO: coversion sin and cosine
                "today": gym.spaces.Box(low=0, high=1),
                # TODO: remember to add seasonality
            }
        )

        # action space is coefficients of the Bernstein polynomial
        self.action_space = gym.spaces.Box(
            low=0,
            high=float("inf"),
            shape=(num_room_categories, NUM_COEFFICIENTS),
            dtype=np.float32,
        )

        # Initiliaze the state
        self.today = start_simulation
        self.start_simulation = start_simulation
        self.end_simulation = end_simulation
        avg_num_requests: int
        self.hotel = hotel

        # Instantiate the request generator and acceptance model
        self.request_generator: RequestGenerator
        self.acceptance_model: SimpleAcceptanceModel
        avg_num_requests = random.randint(MIN_AVG_NUM_REQUESTS, MAX_AVG_NUM_REQUESTS)
        self.request_generator = RequestGenerator(
            avg_num_requests, end_simulation, distribution=None
        )
        self.acceptance_model = SimpleAcceptanceModel()

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

        # Generate prices for each day of valid requests
        # prices = self._generate_prices(valid_requests)
        prices = self._generate_random_prices(valid_requests)

        # Accept or reject the requests
        accepted_mask = self._accept(prices)
        accepted_requests = self._apply_mask(accepted_mask, valid_requests)

        # Calculate the revenue of accepted requests
        reward = self._calculate_reward(prices, accepted_mask)

        # Update the hotel occupancy
        self._update_occupancy(accepted_requests)

        # Update the demand
        self._update_demand(requests)

        # Move to the next day
        self.today += timedelta(days=1)
        done = self.today >= self.end_simulation
        observation = self.observation_space.sample()
        reward = 0  # Placeholder for reward calculation
        info = {}  # Placeholder for additional info
        return observation, reward, done, info

    def reset(self, seed=None) -> tuple:
        """
        Resets the environment to its initial state.

        This method is called to reset the environment to its initial state at the beginning of an episode.
        It also ensures that the randomness and reproducibility are controlled by calling the parent class's reset method.

        Args:
            seed (int, optional): An optional seed for the random number generator to ensure reproducibility. Defaults to None.

        Returns:
            tuple: A tuple containing the initial observation from the observation space and an empty dictionary for additional info.
        """
        super().reset(seed=seed)  # requires this call to control randomness and reproducibility

        # Reset the state of the environment to the initial state
        self.today = self.start_simulation

        # Return the initial observation and info
        return self.observation_space.sample(), {}

    # OTHER FUNCTIONS
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
            if not self.hotel.check_availability(request):
                alternative_room = self._check_alternatives(request)
                if alternative_room is None:
                    masks.append(0)
                else:
                    masks.append(1)
            else:
                masks.append(1)
        return np.array(masks)

    def _check_alternatives(self, request: Request) -> RoomCategory | None:
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

    # TODO: implement this function using Bernstein coeffients
    def _generate_prices(self, valid_requests: np.ndarray):
        pass

    def _generate_random_prices(self, valid_requests: np.ndarray) -> np.ndarray:
        """
        Generate daily random prices for valid requests.

        Parameters:
        valid_requests (np.ndarray): An array of valid requests for which prices need to be generated.

        Returns:
        np.ndarray: An array of randomly generated prices corresponding to each valid request.
        """
        prices = []
        for request in valid_requests:
            total_price = 0
            reservation_days = request.get_total_reservation_days()
            for _ in range(reservation_days):
                rng = np.random.default_rng()
                random_daily_price = rng.uniform(MIN_PRICE, MAX_PRICE)
                total_price += random_daily_price
            prices.append(total_price)
        return np.array(prices)

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

    def _update_occupancy(self, accepted_requests: np.ndarray) -> None:
        """
        Update the occupancy of the hotel based on the accepted requests.

        Parameters:
        accepted_requests (np.ndarray): An array of accepted requests.
        """
        for request in accepted_requests:
            self.hotel.update_occupancy(request)

    def _update_demand(self, requests: np.ndarray) -> None:
        """
        Update the demand for the current day based on the generated requests.

        Parameters:
        requests (np.ndarray): An array of requests generated for the current day.
        """
        pass

    def _accept(self, prices: np.ndarray) -> np.ndarray:
        """
        Accept or reject the proposed prices based on the acceptance model.

        Parameters:
        prices (np.ndarray): An array of prices proposed to the customers.

        Returns:
        np.ndarray: A boolean array indicating which requests were accepted.
        """
        accepted_prices = []
        for price in prices:
            accepted_prices.append(self.acceptance_model.accept(price))
        return np.array(accepted_prices)
