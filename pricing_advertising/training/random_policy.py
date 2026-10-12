"""Evaluate a uniform random policy without training."""

from datetime import date
from typing import Any

import gymnasium as gym
import numpy as np

from pricing_advertising.training.baseline_evaluation import run_baseline_simulation


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
    return run_baseline_simulation(
        lambda action_space: RandomPolicy(action_space, seed),
        model="Random",
        policy_name="UniformRandomPolicy",
        seed=seed,
        eval_seed=eval_seed,
        start_simulation=start_simulation,
        end_simulation=end_simulation,
        num_category1_rooms=num_category1_rooms,
        num_category2_rooms=num_category2_rooms,
        num_category3_rooms=num_category3_rooms,
        eval_episodes=eval_episodes,
        avg_num_requests=avg_num_requests,
        run_name=run_name,
        price_plot_request_dates=price_plot_request_dates,
        allow_category_alternative=allow_category_alternative,
        allow_period_alternative=allow_period_alternative,
    )
