"""Evaluate a uniform random policy without training."""

from datetime import date
from typing import Any

import gymnasium as gym
import mlflow
import numpy as np
from loguru import logger

from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.plots.episode_metrics import EpisodeMetrics
from pricing_advertising.plots.eval_artifact_plots import log_eval_artifact_plots


class RandomPolicy:
    def __init__(self, action_space: gym.spaces.Space, seed: int) -> None:
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("Random pricing evaluation requires a Box action space.")
        self.action_space = action_space
        self.action_space.seed(seed)

    def predict(self, _observation: dict[str, Any]) -> tuple[np.ndarray, None]:
        return self.action_space.sample(), None


# ------------ RUN SIMULATION FUNCTION ------------
def run_random_policy_simulation(
    # ------------ Seed parameters ------------
    seed: int,
    eval_seed: int,
    # ---------- Hotel parameters ----------
    start_simulation: date,
    end_simulation: date,
    num_category1_rooms: int,
    num_category2_rooms: int,
    num_category3_rooms: int,
    # ---------- Evaluation parameters ----------
    eval_episodes: int,
    avg_num_requests: int,
    price_plot_request_dates: list[tuple[int, int, int]] | None = None,
    allow_category_alternative: bool = True,
    allow_period_alternative: bool = True,
    run_name: str = "Random_uniform",
) -> float:
    """Return mean episode revenue in EUR and log evaluation results to MLflow.

    Load the BAR table before calling, as for PPO. ``seed`` controls actions;
    ``eval_seed`` controls the environment and its fixed request scenario.
    Repeated episodes keep those requests and advance both random streams.
    """
    # ------------ Checks ------------
    if isinstance(eval_episodes, bool) or not isinstance(eval_episodes, int) or eval_episodes <= 0:
        raise ValueError("eval_episodes must be a positive integer.")
    for name, value in (("seed", seed), ("eval_seed", eval_seed)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
            raise ValueError(f"{name} must be an integer in [0, 2**32).")
    if (
        isinstance(avg_num_requests, bool)
        or not isinstance(avg_num_requests, int)
        or avg_num_requests <= 0
    ):
        raise ValueError("avg_num_requests must be a positive integer.")
    if start_simulation > end_simulation:
        raise ValueError("start_simulation must not be after end_simulation.")
    snapshot_dates = sorted({date(*value) for value in (price_plot_request_dates or [])})
    if any(not start_simulation <= value <= end_simulation for value in snapshot_dates):
        raise ValueError("Price plot request dates must belong to the simulation period.")

    # ------------ MLflow setup ------------
    mlflow.set_experiment("PricingEnv_Simulation")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": "Random",
                "policy": "UniformRandomPolicy",
                "training_duration": 0,
                "n_envs": 1,
                "reward_normalization": "None",
                "norm_obs": False,
                "norm_reward": False,
                "eval_norm_reward": False,
                "eval_episodes": eval_episodes,
                "seed": seed,
                "eval_request_seed": eval_seed,
                "request_scenario_mode": "fixed_per_environment",
                "start_simulation": start_simulation,
                "end_simulation": end_simulation,
                "num_category1_rooms": num_category1_rooms,
                "num_category2_rooms": num_category2_rooms,
                "num_category3_rooms": num_category3_rooms,
                "avg_num_requests": avg_num_requests,
                "allow_category_alternative": allow_category_alternative,
                "allow_period_alternative": allow_period_alternative,
            }
        )

        # ------------ Hotel and evaluation environment ------------
        hotel = Hotel(
            num_category1_rooms=num_category1_rooms,
            num_category2_rooms=num_category2_rooms,
            num_category3_rooms=num_category3_rooms,
        )
        env = PricingEnv(
            start_simulation=start_simulation,
            end_simulation=end_simulation,
            hotel=hotel,
            avg_num_requests=avg_num_requests,
            seed=eval_seed,
            collect_eval_artifacts=True,
            artifact_snapshot_dates=snapshot_dates,
            allow_category_alternative=allow_category_alternative,
            allow_period_alternative=allow_period_alternative,
        )

        try:
            # ------------ Policy setup (no training) ------------
            policy = RandomPolicy(env.action_space, seed)
            metrics = EpisodeMetrics(split="val")
            episode_rewards: list[float] = []
            episode_lengths: list[int] = []

            # ------------ Evaluation ------------
            for episode_id in range(eval_episodes):
                observation, _ = env.reset(seed=eval_seed if episode_id == 0 else None)
                episode_reward = 0.0
                steps: list[dict[str, Any]] = []
                terminated = truncated = False

                while not (terminated or truncated):
                    action, _ = policy.predict(observation)
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_reward += float(reward)
                    metrics.record_day(info["step_metrics"])
                    payload = info.get("eval_artifacts")
                    if payload is None:
                        raise RuntimeError("Missing eval_artifacts data in random evaluation.")
                    steps.append(payload)

                metrics.end_episode()
                episode_rewards.append(episode_reward)
                episode_lengths.append(len(steps))
                mlflow.log_metric("eval/episode_reward", episode_reward, step=episode_id)
                mlflow.log_metric("eval/episode_length", len(steps), step=episode_id)

                # Checkpoint zero denotes a policy with no training steps.
                artifact = {
                    "schema_version": 3,
                    "decision_end_inclusive": True,
                    "checkpoint_timesteps": 0,
                    "env_index": 0,
                    "episode_id": episode_id,
                    "snapshot_dates": [value.isoformat() for value in snapshot_dates],
                    "steps": steps,
                }
                mlflow.log_dict(
                    artifact,
                    f"eval_data/00000000_timesteps/env_00/episode_{episode_id:05d}.json",
                )
                log_eval_artifact_plots(artifact)
                logger.info(
                    "Random evaluation episode {}/{}: reward={:.2f} EUR",
                    episode_id + 1,
                    eval_episodes,
                    episode_reward,
                )

            # ------------ Final results ------------
            # Match PPO's metric names, with zero training timesteps for this baseline.
            metrics.log_step_metrics(step=0)
            metrics.finalize()
            mean_reward = float(np.mean(episode_rewards))
            std_reward = float(np.std(episode_rewards))
            mlflow.log_metric("final_eval_mean_reward", mean_reward)
            mlflow.log_metric("final_eval_std_reward", std_reward)
            mlflow.log_dict(
                {
                    "seed": seed,
                    "eval_request_seed": eval_seed,
                    "episode_rewards": episode_rewards,
                    "episode_lengths": episode_lengths,
                },
                "eval_data/results.json",
            )
            return mean_reward
        finally:
            env.close()
