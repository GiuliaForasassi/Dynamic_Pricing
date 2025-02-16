from datetime import date

from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel


def run_test_simulation() -> None:
    # Define simulation period
    start_simulation = date(2025, 1, 1)
    end_simulation = date(2025, 1, 31)

    # Initialize the hotel
    hotel = Hotel.create_hotel(100, 20, 30, 50)
    env = PricingEnv(start_simulation, end_simulation, 3, hotel)

    # Run the simulation
    state = env.reset()
    total_reward = 0
    done = False

    print("Starting simulation...")
    while not done:
        action = env.policy.select_price(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        print(f"Day: {env.today} | Revenue: {reward} | Total Revenue: {total_reward}")

    print("Simulation completed.")
    print(f"Total revenue over {end_simulation - start_simulation} days: {total_reward}")
