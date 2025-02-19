from datetime import date

from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel


def run_test_simulation() -> None:
    # Define simulation period
    start_simulation = date(2025, 1, 1)
    end_simulation = date(2025, 1, 31)
    simulation_days = (end_simulation - start_simulation).days

    # Initialize the hotel
    hotel = Hotel.create_hotel(100, 20, 30, 50)
    env = PricingEnv(start_simulation, end_simulation, 3, hotel)

    observation, info = env.reset(seed=42)

    # Run the simulation for 14 days ??
    for _ in range(simulation_days):
        action = env.policy.select_price(observation)
        observation, reward, done, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset(seed=42)
    env.close()
    print("Simulation completed")
