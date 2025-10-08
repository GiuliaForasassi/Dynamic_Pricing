import sys
from datetime import date

import mlflow
import numpy as np
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.vec_env import SubprocVecEnv

import pricing_advertising.environment.__init__  # noqa: F401

# import i tuoi oggetti
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.utils.constants import (
    N_ENVS,
    SEEDS,
)
from pricing_advertising.utils.mlflow_logger import MLflowOutputFormat


def run_test_simulation() -> float:
    # Fixed hyperparameters
    timesteps = 5000
    learning_rate = 1e-4
    gamma = 0.99

    # Setup MLflow
    mlflow.set_experiment("PricingEnv_Simulation")

    with mlflow.start_run(run_name="FixedParams_Run"):
        # Define simulation period
        start_simulation = date(2025, 1, 1)
        end_simulation = date(2025, 12, 1)

        # Define hotel parameters
        num_category1_rooms = 10
        num_category2_rooms = 20
        num_category3_rooms = 30

        seed = SEEDS[0]

        # Log the hyperparameters
        mlflow.log_params(
            {
                "seed": seed,
                "start_simulation": start_simulation,
                "end_simulation": end_simulation,
                "num_category1_rooms": num_category1_rooms,
                "num_category2_rooms": num_category2_rooms,
                "num_category3_rooms": num_category3_rooms,
                "model": "PPO",
                "policy": "MlpPolicy",
                "timesteps": timesteps,
                "learning_rate": learning_rate,
                "gamma": gamma,
            }
        )

        # Logger setup
        loggers = Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
        )

        # Initialize the hotel
        hotel = Hotel(
            num_category1_rooms=num_category1_rooms,
            num_category2_rooms=num_category2_rooms,
            num_category3_rooms=num_category3_rooms,
        )

        kwargs = {
            "seed": seed,
            "start_simulation": start_simulation,
            "end_simulation": end_simulation,
            "hotel": hotel,
        }

        vec_env = make_vec_env(
            "PricingEnv-v0",
            n_envs=1,
            env_kwargs=kwargs,
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "fork"},
        )

        model = PPO(
            "MultiInputPolicy", vec_env, verbose=1, learning_rate=learning_rate, gamma=gamma
        )
        model.set_logger(loggers)
        model.learn(total_timesteps=timesteps, log_interval=1, progress_bar=True)

        observations = vec_env.reset()
        total_rewards = np.zeros(N_ENVS)

        done = np.zeros(N_ENVS, dtype=bool)
        while not np.all(done):
            actions, _ = model.predict(observations, deterministic=True)  # type: ignore
            observations, rewards, done, info = vec_env.step(actions)
            total_rewards += rewards

        vec_env.close()
        average_reward = float(np.mean(total_rewards))
        logger.info(f"Total rewards: {total_rewards}")
        logger.info(f"Average reward: {average_reward}")
        logger.info("Simulation finished.")

        # Log metrics
        mlflow.log_metric("avg_reward", average_reward)

        return average_reward


if __name__ == "__main__":
    avg_reward = run_test_simulation()
    print(f"Simulation completed. Average reward: {avg_reward}")
