import gc
import sys
from datetime import date
from typing import Any

import mlflow
import numpy as np
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.vec_env import SubprocVecEnv

import pricing_advertising.environment.__init__  # noqa: F401

# your imports
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.plots.plot_callback import PlotCallback
from pricing_advertising.utils.constants import SEEDS
from pricing_advertising.utils.mlflow_logger import MLflowOutputFormat


def run_simulation(
    # ======= TRAINING SCHEDULE =======
    # total training = total_cycles * steps_per_cycle
    # total_cycles --> only for “scheduling” to break the training: for doing log/metrics
    total_cycles: int,
    # steps_per_cycle: multiple of n_steps * n_envs
    steps_per_cycle: int,
    # ======= PPO/ENV =======
    n_envs: int,
    # n_steps: timesteps per environment in the rollout
    n_steps: int,
    learning_rate: float,
    gamma: float,
    # epochs: number of optimisation passes on the same rollout
    n_epochs: int,
    policy: str = "MultiInputPolicy",
    # ======= EVAL =======
    eval_episodes: int = 5,
    # ======= SIMULAZIONE HOTEL =======
    start_simulation: date = date(2025, 1, 1),
    end_simulation: date = date(2025, 12, 1),
    num_category1_rooms: int = 50,
    num_category2_rooms: int = 50,
    num_category3_rooms: int = 50,
    # ======= SEED =======
    seed: int = SEEDS[0],
) -> float:
    # ============ MLFLOW SETUP ============
    mlflow.set_experiment("PricingEnv_Simulation")
    with mlflow.start_run(run_name="PPO_training"):
        # ============ Logger the hyperparameters ============
        mlflow.log_params(
            {
                "model": "PPO",
                "policy": policy,
                "total_cycles": total_cycles,
                "steps_per_cycle": steps_per_cycle,
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

        vec_env = make_vec_env(
            "PricingEnv-v0",
            n_envs=n_envs,
            env_kwargs=env_kwargs,
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "fork"},
        )

        # another env for evaluation (n_envs=1)
        eval_env = make_vec_env(
            "PricingEnv-v0",
            n_envs=1,
            env_kwargs=env_kwargs,
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "fork"},
        )

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

        # ============ Callback ============
        callback = PlotCallback(log_to_mlflow=True)

        # ============ Training ============
        total_trained: int = 0
        last_mean_r: float = float("nan")
        last_std_r: float = float("nan")

        for _ in range(total_cycles):
            # --- Train on the same model ---
            model.learn(
                total_timesteps=steps_per_cycle,
                reset_num_timesteps=False,
                callback=callback,
                log_interval=1,
                progress_bar=True,
            )
            total_trained += steps_per_cycle

            # --- Evaluation: here you are doing the prediction ---
            mean_r, std_r = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=eval_episodes,
                deterministic=True,
                return_episode_rewards=False,
            )
            last_mean_r, last_std_r = np.asarray(mean_r).item(), np.asarray(std_r).item()

            # log on MLflow (cumulated timesteps)
            mlflow.log_metrics(
                {
                    "eval_mean_reward": last_mean_r,
                    "eval_std_reward": last_std_r,
                },
                step=total_trained,
            )

        # ============ Cleanup ============
        vec_env.close()
        eval_env.close()
        del model, vec_env, eval_env
        gc.collect()

        # ============ Log metrics ============
        mlflow.log_metric("final_eval_mean_reward", last_mean_r)
        mlflow.log_metric("final_eval_std_reward", last_std_r)

        return last_mean_r
