import gc
import sys
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import mlflow
import numpy as np
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
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
    # ======= OPTIONAL PARAMETERS =======
    eval_frequency: int,
    price_plot_request_dates: list[tuple[int, int, int]] | None = None,
    avg_num_requests: int = 250,
    eval_seed: int | None = None,
) -> float:
    # ============ SEED CHOISE ============
    if n_envs <= 0:
        raise ValueError("n_envs deve essere positivo.")
    if avg_num_requests <= 0:
        raise ValueError("avg_num_requests deve essere positivo.")

    # Un calendario di richieste distinto per ogni ambiente di training.
    train_request_seeds = [seed + index for index in range(n_envs)]

    # Per default usiamo il primo seed successivo a quelli del training.
    if eval_seed is None:
        eval_seed = seed + n_envs

    if eval_seed in train_request_seeds:
        raise ValueError("Il seed di validation deve essere distinto da quelli del training.")

    # Compatibilità con i generatori e le librerie che usano seed a 32 bit.
    if any(value < 0 or value >= 2**32 for value in [*train_request_seeds, eval_seed]):
        raise ValueError("I seed devono appartenere all'intervallo [0, 2**32).")

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
                "avg_num_requests": avg_num_requests,
                "train_request_seeds": str(train_request_seeds),
                "eval_request_seed": eval_seed,
                "request_scenario_mode": "fixed_per_environment",
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
            start_simulation=start_simulation,
            end_simulation=end_simulation,
            hotel=hotel,
            avg_num_requests=avg_num_requests,
        )

        # Il seed va passato al COSTRUTTORE:
        # le richieste vengono pregenerate durante __init__.
        train_factories = [
            partial(
                gym.make,
                "PricingEnv-v0",
                seed=request_seed,
                **env_kwargs,
            )
            for request_seed in train_request_seeds
        ]

        vec_env = VecMonitor(
            SubprocVecEnv(
                train_factories,
                start_method="fork",
            )
        )
        vec_env.seed(seed)

        eval_env_kwargs = {
            **env_kwargs,
            "collect_eval_artifacts": True,
            "artifact_snapshot_dates": [date(*value) for value in (price_plot_request_dates or [])],
        }

        eval_env = VecMonitor(
            SubprocVecEnv(
                [
                    partial(
                        gym.make,
                        "PricingEnv-v0",
                        seed=eval_seed,
                        **eval_env_kwargs,
                    )
                ],
                start_method="fork",
            )
        )
        eval_env.seed(eval_seed)

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
            eval_freq=eval_frequency,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
            best_model_save_path="models/best_model/",
            log_path="models/eval_logs/",
            log_to_mlflow=True,
        )

        # ============ List of callbacks ============
        callback = CallbackList(
            [
                plot_callback,
                eval_callback,
            ]
        )

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
