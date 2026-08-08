import gc
import sys
from datetime import date
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

import pricing_advertising.environment.__init__  # noqa: F401

# your imports
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.plots.plot_callback import EvalOccupancyCallback, PlotCallback
from pricing_advertising.utils.constants import SEEDS
from pricing_advertising.utils.mlflow_logger import MLflowOutputFormat


def run_simulation(
    # ======= TRAINING SCHEDULE =======
    total_timesteps: int,
    # ======= PPO/ENV =======
    n_envs: int,
    # n_steps: timesteps per environment in the rollout
    n_steps: int,
    learning_rate: float,
    gamma: float,
    # epochs: number of optimization passes on the same rollout
    n_epochs: int,
    policy: str,
    # ======= EVAL =======
    eval_episodes: int,
    # ======= SIMULAZIONE HOTEL =======
    start_simulation: date,
    end_simulation: date,
    num_category1_rooms: int,
    num_category2_rooms: int,
    num_category3_rooms: int,
    # ======= SEED =======
    seed: int,
) -> float:
    # ============ MLFLOW SETUP ============
    mlflow.set_experiment("PricingEnv_Simulation")
    with mlflow.start_run(run_name="PPO_training"):
        # ============ Logger the hyperparameters ============
        mlflow.log_params(
            {
                "model": "PPO",
                "policy": policy,
                "training_duration": total_timesteps,
                "n_envs": n_envs,
                "n_steps": n_steps,
                "n_epochs": n_epochs,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "eval_episodes": eval_episodes,
                "seed": seed,
                "start_simulation": start_simulation,
                "end_simulation": end_simulation,
                "num_category1_rooms": num_category1_rooms,
                "num_category2_rooms": num_category2_rooms,
                "num_category3_rooms": num_category3_rooms,
            }
        )

        # ============ Logger setup ============
        logger = Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
        )

        # ============ Construct the hotel and env ============
        hotel = Hotel(
            num_category1_rooms=num_category1_rooms,
            num_category2_rooms=num_category2_rooms,
            num_category3_rooms=num_category3_rooms,
        )

        env_kwargs: dict[str, Any] = dict(
            seed=seed,
            start_simulation=start_simulation,
            end_simulation=end_simulation,
            hotel=hotel,
        )

        # da passare a PPO
        vec_env = make_vec_env(
            "PricingEnv-v0",
            n_envs=n_envs,
            env_kwargs=env_kwargs,
            # vec_env_cls=DummyVecEnv,
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "fork"},
        )
        # use Moniotr/VecMonitor to ensure that EvalCallback logs correctly
        vec_env = VecMonitor(vec_env)

        # da passare a eval_callback
        eval_env = make_vec_env(
            "PricingEnv-v0",
            n_envs=1,
            env_kwargs=env_kwargs,
            # vec_env_cls=DummyVecEnv,
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "fork"},
        )

        eval_env = VecMonitor(eval_env)

        # ============ Model ============
        # PPO works into rollout of size = n_steps * n_envs
        model = PPO(
            policy,
            vec_env,
            verbose=1,
            learning_rate=learning_rate,
            gamma=gamma,
            n_epochs=n_epochs,
            n_steps=n_steps,
            seed=seed,
        )
        model.set_logger(logger)

        # ============ Callback custom for plots ============
        plot_callback = PlotCallback(
            log_to_mlflow=True,
            track_env_idx=0,
        )

        # ============ Callback di valutazione ============
        eval_callback = EvalOccupancyCallback(
            eval_env,
            eval_freq=500,  # todo: RIVEDI: valutare in base a num di env di training
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
            best_model_save_path="models/best_model/",
            log_path="models/eval_logs/",
            log_to_mlflow=True,
        )

        # ============ List of callbacks ============
        callback = CallbackList([plot_callback, eval_callback])

        # --- Train on the same model ---
        model.learn(
            total_timesteps=total_timesteps, callback=callback, log_interval=1, progress_bar=True
        )

        # --- Evaluation: here you are doing the prediction ---
        # Look for the best model if exists
        best_model_dir = Path("models/best_model")
        best_model_path = best_model_dir / "best_model.zip"

        # Ricarico il modello migliore per valutarlo, altrimenti uso l'ultimo modello
        best_model = PPO.load(best_model_path, env=eval_env) if best_model_path.exists() else model

        # Evaluation of the best model on eval_env
        mean_reward, std_reward = evaluate_policy(
            best_model,
            eval_env,
            n_eval_episodes=eval_episodes,
            deterministic=True,
        )

        if isinstance(mean_reward, list) or isinstance(std_reward, list):
            raise ValueError("Unexpected type list")

        # Log on MLflow
        mlflow.log_metric("final_eval_mean_reward", float(mean_reward))
        mlflow.log_metric("final_eval_std_reward", float(std_reward))

        # ============ Cleanup ============
        vec_env.close()
        eval_env.close()

        return mean_reward
