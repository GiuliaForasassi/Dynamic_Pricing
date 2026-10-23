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
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

import pricing_advertising.environment.__init__  # noqa: F401

# your imports
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.plots.plot_callback import EvalOccupancyCallback, PlotCallback
from pricing_advertising.training.baseline_returns import evaluate_baseline_returns
from pricing_advertising.utils.constants import SEEDS
from pricing_advertising.utils.mlflow_logger import MLflowOutputFormat


# ------------ RUN SIMULATION FUNCTION ------------
def run_simulation(
    # ---------- PARAMETERS ------------------
    # ------------ Seed parameters ----------
    seed: int,
    # ---------- Hotel parameters ----------
    start_simulation: date,
    end_simulation: date,
    num_category1_rooms: int,
    num_category2_rooms: int,
    num_category3_rooms: int,
    # --------- PPO and env parameters ----------
    total_timesteps: int,  # Total number of training timesteps
    n_envs: int,  # Number of parallel environments
    n_steps: int,  # Number of steps to run for each environment per update
    learning_rate: float,  # Learning rate for the optimizer
    gamma: float,  # Discount factor for future rewards
    n_epochs: int,  # Number of optimization passes on the same rollout
    policy: str,  # Policy architecture for the PPO agent
    # ------------ Eval parameters ----------
    eval_episodes: int,  # Number of episodes to run during evaluation
    eval_frequency: int,  # Frequency (in timesteps) at which to run evaluation
    eval_seed: int | None = None,
    # ---------- Other parameters ----------
    avg_num_requests: int = 250,  # Average number of requests per day
    price_plot_request_dates: list[tuple[int, int, int]]
    | None = None,  # List of dates for which to plot price requests
    allow_category_alternative: bool = True,
    allow_period_alternative: bool = True,
    # ---------- Additional PPO hyperparameters ----------
    batch_size: int = 64,
    ent_coef: float = 0.0,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    net_arch: list[int] | dict[str, list[int]] | None = None,
    run_name: str = "PPO_vec_normalize",
    evaluate_baselines: bool = True,  # Draw constant/always-increment/random returns in the eval plot
) -> float:
    # --------- Checks ----------------
    if n_envs <= 0:
        raise ValueError("n_envs deve essere positivo.")
    if avg_num_requests <= 0:
        raise ValueError("avg_num_requests deve essere positivo.")

    # -------------- Seeds setup ----------------
    # Each training environment gets a distinct request seed
    # Generate a list of seeds for each training environment
    train_request_seeds = [seed + index for index in range(n_envs)]

    # If no evaluation seed is provided, use the first seed after the training seeds
    if eval_seed is None:
        eval_seed = seed + n_envs

    # Ensure the evaluation seed is distinct from the training seeds
    if eval_seed in train_request_seeds:
        raise ValueError("The evaluation seed must be distinct from the training seeds.")

    # Compatibility with generators and libraries that use 32-bit seeds.
    if any(value < 0 or value >= 2**32 for value in [*train_request_seeds, eval_seed]):
        raise ValueError("Seeds must belong to the interval [0, 2**32).")

    # ------------ MLFLOW SETUP ------------
    mlflow.set_experiment("PricingEnv_Simulation")
    with mlflow.start_run(run_name=run_name) as run:
        # Define the output directory for saving final model, best model and evaluation logs
        output_dir = Path("models") / "vec_normalize" / run.info.run_id
        best_model_dir = output_dir / "best_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir.resolve()}")
        # ----------- Log normalization parameters -----------
        mlflow.log_params(
            {
                "reward_normalization": "VecNormalize",
                "norm_obs": False,
                "norm_reward": True,
                "clip_reward": 10.0,
                "normalization_gamma": gamma,
                "eval_norm_reward": False,
            }
        )
        # ---------- Log model hyperparameters ----------
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
                "batch_size": batch_size,
                "ent_coef": ent_coef,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range,
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
                "allow_category_alternative": allow_category_alternative,
                "allow_period_alternative": allow_period_alternative,
            }
        )

        # ------------ Logger setup ------------
        logger = Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
        )

        # ------------ Hotel setup ------------------
        hotel = Hotel(
            num_category1_rooms=num_category1_rooms,
            num_category2_rooms=num_category2_rooms,
            num_category3_rooms=num_category3_rooms,
        )

        # ------------ Training environment setup ------------------
        env_kwargs: dict[str, Any] = dict(
            start_simulation=start_simulation,
            end_simulation=end_simulation,
            hotel=hotel,
            avg_num_requests=avg_num_requests,
            allow_category_alternative=allow_category_alternative,
            allow_period_alternative=allow_period_alternative,
        )

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
        vec_env = VecNormalize(
            vec_env, norm_obs=False, norm_reward=True, gamma=gamma, clip_reward=10.0
        )

        # ------------ Evaluation environment setup ------------------
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
        eval_env = VecNormalize(
            eval_env, training=False, norm_obs=False, norm_reward=False, gamma=gamma
        )

        # ------------ Model setup ------------------
        # PPO works into rollout of size = n_steps * n_envs
        model = PPO(
            policy,
            vec_env,
            verbose=1,
            learning_rate=learning_rate,
            gamma=gamma,
            n_epochs=n_epochs,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            policy_kwargs={"net_arch": net_arch} if net_arch is not None else None,
            seed=seed,
        )
        mlflow.log_params({"net_arch": str(model.policy.net_arch)})
        model.set_logger(logger)

        # ------------ Callback custom for plots ------------------
        # Plot callback for visualizing training progress
        plot_callback = PlotCallback(
            log_to_mlflow=True,
            track_env_idx=0,
        )

        # Evaluation callback for evaluating the model
        baseline_returns = (
            evaluate_baseline_returns(
                env_kwargs, eval_seed=eval_seed, episodes=eval_episodes, seed=seed
            )
            if evaluate_baselines
            else None
        )
        eval_callback = EvalOccupancyCallback(
            eval_env,
            eval_freq=eval_frequency,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
            best_model_save_path=str(best_model_dir),
            log_path=str(output_dir / "eval_logs"),
            log_to_mlflow=True,
            baseline_returns=baseline_returns,
        )

        # ------------ List of callbacks ------------------
        callback = CallbackList(
            [
                plot_callback,
                eval_callback,
            ]
        )

        # ------------ Training ------------------
        # Start the training process
        model.learn(
            total_timesteps=total_timesteps, callback=callback, log_interval=1, progress_bar=True
        )
        # Save the final model and the VecNormalize statistics
        model.save(output_dir / "final_model.zip")
        vec_env.save(str(output_dir / "final_vecnormalize.pkl"))

        # --- Evaluation: here you are doing the prediction ---
        # Look for the best model if exists
        best_model_path = best_model_dir / "best_model.zip"

        # Load the best model if it exists, otherwise use the last trained model
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

        # ------------ Cleanup ------------------
        vec_env.close()
        eval_env.close()

        return mean_reward
