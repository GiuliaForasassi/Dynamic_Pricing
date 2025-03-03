"""
from datetime import date

from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.policies.pricing.random_policy import RandomPricingPolicy

def run_test_simulation() -> None:
    # Define simulation period
    start_simulation = date(2025, 1, 1)
    end_simulation = date(2025, 1, 31)
    simulation_days = (end_simulation - start_simulation).days

    # Initialize the hotel
    hotel = Hotel(
        num_total_rooms=10,
        num_category1_rooms=3,
        num_category2_rooms=4,
        num_category3_rooms=3,
        start_simulation=start_simulation,
        end_simulation=end_simulation,
    )
    env = PricingEnv(start_simulation, end_simulation, hotel)
    policy = RandomPricingPolicy(start_simulation, end_simulation, num_coefficients=4)

    observation, info = env.reset(seed=42)

    for _ in range(simulation_days):
        action = policy.select_price(observation)
        observation, reward, done, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()
    print("Simulation completed")"""
