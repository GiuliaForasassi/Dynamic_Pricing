"""Evaluate a baseline that never raises the BAR level (prices stay at the initial level)."""

from datetime import date
from typing import Any

import gymnasium as gym
import numpy as np

from pricing_advertising.training.baseline_evaluation import run_baseline_simulation


class ConstantBarPolicy:
    """Zero Bernstein coefficients: the BAR increment probability is always 0."""

    def __init__(self, action_space: gym.spaces.Space) -> None:
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("Constant BAR evaluation requires a Box action space.")
        self.action = np.zeros(action_space.shape, dtype=action_space.dtype)

    def predict(self, _observation: dict[str, Any]) -> tuple[np.ndarray, None]:
        return self.action.copy(), None


# ------------ RUN SIMULATION FUNCTION ------------
def run_constant_bar_policy_simulation(
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
    run_name: str = "Constant_bar",
) -> float:
    """Return mean episode revenue in EUR and log evaluation results to MLflow.

    The BAR level is never incremented, so it stays constant from start to end.
    The environment forces the increment probability to 0 and ignores the action.
    ``eval_seed`` controls the environment and its fixed request scenario.
    """
    return run_baseline_simulation(
        lambda action_space: ConstantBarPolicy(action_space),
        model="ConstantBar",
        policy_name="NeverIncrementPolicy",
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
        bar_update_mode="never",
    )
