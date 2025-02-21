import random
from datetime import date, timedelta
from typing import ClassVar

import gymnasium as gym
import numpy as np

from pricing_advertising.clients.client_generator import ClientGenerator
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.hotel.room import RoomCategory
from pricing_advertising.policies.pricing.random_policy import RandomPricingPolicy

# Register the PricingEnv environment with OpenAI Gym
# gym.register(id="PricingEnv-v0", entry_point="pricing_advertising.environment.env:PricingEnv")


# Implement the PricingEnv environment as a subclass of gym.Env
class PricingEnv(gym.Env):
    # Metadata is a required attribute for the environment
    metadata: ClassVar[dict] = {"render.modes": ["human"], "render_fps": 1}

    def __init__(self, start_simulation: date, end_simulation: date, hotel: Hotel) -> None:
        super().__init__()

        # Define the dimensions of the state and action spaces and other vars/structure data you need
        num_room_categories = len(RoomCategory)
        num_coefficients = 4
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
            shape=(num_room_categories, num_coefficients),
            dtype=np.float32,
        )

        # Initiliaze the state
        self.today: date = start_simulation
        self.start_simulation: date = start_simulation
        self.end_simulation: date = end_simulation
        self.hotel: Hotel = hotel

        # Initialize the revenue

    # STEP FUNCTION
    # Execute 1 time step within the environment
    def step(self, action: gym.spaces.Box):
        # If no acions is provided, sample one from the policy

        # here use bernstein coefficients
        # theta = ....
        #

        # Generate clients
        num_clients = random.randint(4, 15)
        client_generator = ClientGenerator(num_clients, self.end_simulation)
        clients = client_generator.generate_clients(self.today)

        # Generate the price vector using the Bernstein coefficients
        # TODO: dati i client, genera il vettore dei prezzi
        # prices = self.generate_price_vector(theta, len(clients))
        # TODO: booking attempt

        # Client booking process

        # accepted_clients = .....
        # TODO: calcolo reward
        reward = ...
        # TODO: aggiorno info: tempo, occupancy ecc

        # Move to the next day
        self.today += timedelta(days=1)
        done = self.today >= self.end_simulation
        # return observation, reward, info
        return self.observation_space.sample, reward, done, {}

    # RESET FUNCTION
    # Reset the environment to its initial state and return the initial observations
    def reset(self, seed=None):
        super().reset(seed=seed)  # requires this call to control randomness and reproducibility
        # TODO: reset occupancy hotel
        # TODO: reset demand
        # TODO: reset time horizon
        # TODO: inizializzare reward neg a quelli che cancellano
        # return observation, info
        pass

    # RENDER FUNCTION
    # Render the environment
    def render(self):
        pass

    # CLOSE FUNCTION
    # Close the environment
    def close(self):
        pass

    # OTHER FUNCTIONS

    # Translate the environment's state into an observation
    def get_obs(self):
        occupancy = self.hotel.get_occehupancy()

    def get_info(self):
        pass
