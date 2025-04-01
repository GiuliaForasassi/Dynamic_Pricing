"""import sys
from datetime import date

import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.utils.mlflow_logger import MLflowOutputFormat

# Assuming that the simulation start the first of the month


def run_test_simulation() -> None:
    # Set experiments ahd start MLflow run
    mlflow.set_experiment("PricingEnv_Simulation")

    with mlflow.start_run(run_name="PricingEnv_TestRun"):
        # Define simulation period
        start_simulation = date(2025, 1, 1)
        end_simulation = date(2025, 1, 31)
        simulation_days = (end_simulation - start_simulation).days

        # Log the hyperparameters
        mlflow.log_params(
            {
                "start_simulation": start_simulation,
                "end_simulation": end_simulation,
                "hotel_rooms": {
                    "num_category1_rooms": 3,
                    "num_category2_rooms": 4,
                    "num_category3_rooms": 3,
                },
                "timesteps": 1000,
                "model": "PPO",
                "policy": "MlpPolicy",
            }
        )

        # Logger setup
        loggers = Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
        )

        # Initialize the hotel
        hotel = Hotel(
            num_category1_rooms=3,
            num_category2_rooms=4,
            num_category3_rooms=3,
        )
        env = PricingEnv(start_simulation, end_simulation, hotel)
        vec_env = DummyVecEnv([lambda: env])

        model = PPO("MlpPolicy", vec_env, verbose=1)
        model.set_logger(loggers)
        model.learn(total_timesteps=1000, log_interval=1, progress_bar=True)

        observation = vec_env.reset()
        total_reward = 0

        for _ in range(simulation_days):
            action, _ = model.predict(observation, deterministic=True)  # fix it
            observation, reward, done, info = vec_env.step(action)

            total_reward += reward[0]

            if done[0]:
                observation = vec_env.reset()

        vec_env.close()
        print("Simulation completed")

        # Log metrics
        mlflow.log_metric("total_reward", total_reward)

        # Log a basic artifact
        with open("simulation_output.txt", "w") as f:
            f.write(f"Total reward: {total_reward}")
        mlflow.log_artifact("simulation_output.txt")"""
