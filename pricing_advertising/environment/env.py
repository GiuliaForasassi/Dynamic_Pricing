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
    MIN_AVG_NUM_REQUESTS,
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
        request_generator: RequestGenerator
        acceptance_model: SimpleAcceptanceModel
        avg_num_requests = random.randint(MIN_AVG_NUM_REQUESTS, MAX_AVG_NUM_REQUESTS)
        request_generator = RequestGenerator(
            avg_num_requests, end_simulation, distribution=None
        )  # Generator of requests
        acceptance_model = SimpleAcceptanceModel()  # Acceptance model

    def step(self, action):
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
