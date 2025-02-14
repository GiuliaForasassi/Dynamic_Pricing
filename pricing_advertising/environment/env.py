from datetime import date, timedelta

import gymnasium as gym
import numpy as np

from pricing_advertising.clients.client_generator import ClientGenerator
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.policies.random_policy import RandomPolicy


class PricingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, start_simulation: date, end_simulation: date, num_coefficients: int, hotel: Hotel
    ) -> None:
        super().__init__()

        # Define the action and observation space
        self.num_coefficients = num_coefficients  # number of bernstein coefficients
        self.action_space = gym.spaces.Box(
            low=0, high=200, shape=(self.num_coefficients,), dtype=np.float32
        )  # action space is the vector of bernstein coefficients
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(...), dtype=np.float32)
        # TODO: demand, booking lead time, seasonability
        # state variables:
        # - occupancy of the hotel
        # - demand of the clients
        # - seasonability

        # Initiliaze the state
        self.today: date = start_simulation
        self.start_simulation: date = start_simulation
        self.end_simulation: date = end_simulation

        # Initialize the random policy
        self.policy = RandomPolicy(self.action_space)

        # Initialize the revenue

    # STEP FUNCTION
    # Execute 1 time step within the environment
    def step(self, action, num_clients: int):
        # If no acions is provided, sample one from the policy

        # here use berstain coefficients
        # theta = ....
        #

        # Generate clients
        self.num_clients = num_clients
        client_generator = ClientGenerator(num_clients, self.end_simulation)
        clients = client_generator.generate_clients(self.today)

        # Generate the price vector using the Bernstein coefficients
        # TODO: dati i client, genera il vettore dei prezzi
        # prices = self.generate_price_vector(theta, len(clients))
        # TODO: booking attempt

        # Client booking process
        # for clients, prices in zip(clients, prices):

        # accepted_clients = .....
        # TODO: calcolo reward
        # reward = ......
        # TODO: aggiorno info: tempo, occupancy ecc

        # Move to the next day
        self.today += timedelta(days=1)
        done = self.today >= self.end_simulation
        # return observation, reward, info
        return self.observation_space.sample, reward, done, {}

    # RESET FUNCTION
    # Reset the environment to its initial state and return the initial observations
    def reset(self):
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
