from datetime import date, timedelta

import gymnasium as gym
import numpy as np

from pricing_advertising.clients.generation import ClientGenerator


class PricingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, start_simulation: date, end_simulation: date) -> None:
        super().__init__()

        # Define the action and observation space

        # Initiliaze the state
        self.today: date = start_simulation
        self.start_simulation: date = start_simulation
        self.end_simulation: date = end_simulation

    # STEP FUNCTION
    # Execute 1 time step within the environment
    def step(self, action):
        # here use berstain coefficients
        # clients generation
        client_generator = ClientGenerator(100, self.end_simulation)
        clients = client_generator.generate_clients(self.today)
        # TODO: dati i client, genera il vettore dei prezzi
        # TODO: acceptance function semplice
        # TODO: calcolo reward
        # TODO: aggiorno info: tempo, occupancy ecc
        # go to next day
        self.today += timedelta(days=1)
        # return observation, reward, info
        pass

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
