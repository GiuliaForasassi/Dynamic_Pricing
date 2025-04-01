import sys
from datetime import date

import mlflow
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.vec_env import SubprocVecEnv

from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.utils.constants import N_ENVS
from pricing_advertising.utils.mlflow_logger import MLflowOutputFormat

# Assuming that the simulation start the first of the month


def run_test_simulation() -> None:
    # Set experiments ahd start MLflow run
    mlflow.set_experiment("PricingEnv_Simulation")

    with mlflow.start_run(run_name="PricingEnv_TestRun"):
        # Define simulation period
        start_simulation = date(2025, 1, 1)
        end_simulation = date(2025, 1, 31)

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
        kwargs = {
            "start_simulation": start_simulation,
            "end_simulation": end_simulation,
            "hotel": hotel,
        }
        vec_env = make_vec_env(
            "DynamicPricing-v0",
            n_envs=N_ENVS,
            env_kwargs=kwargs,
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "fork"},
        )

        model = PPO("MlpPolicy", vec_env, verbose=1)
        model.set_logger(loggers)
        model.learn(total_timesteps=1000, log_interval=1, progress_bar=True)

        observations = vec_env.reset()
        total_rewards = np.zeros(N_ENVS)

        done = np.zeros(N_ENVS, dtype=bool)
        while not done[0]:  # check if all have the same time horizon
            actions, _ = model.predict(observations, deterministic=True)  # type: ignore
            observations, rewards, done, info = vec_env.step(actions)

            total_rewards += rewards

        vec_env.close()
        print("Simulation completed")

        # Log metrics
        # mlflow.log_metric("total_reward", total_reward)

        # Log a basic artifact
        with open("simulation_output.txt", "w") as f:
            f.write(f"Total reward: {total_rewards}")
        mlflow.log_artifact("simulation_output.txt")
