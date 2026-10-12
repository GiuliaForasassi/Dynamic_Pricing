"""Evaluate a baseline that raises the BAR level by one at every step."""

from datetime import date
from typing import Any

import gymnasium as gym
import numpy as np

from pricing_advertising.training.baseline_evaluation import run_baseline_simulation


class AlwaysIncrementPolicy:
    """Placeholder action: the environment forces the increment probability to 1.

    The Bernstein parametrisation cannot reach probability 1 at low occupancy,
    so the action is ignored when the environment uses ``bar_update_mode="always"``.
    """

    def __init__(self, action_space: gym.spaces.Space) -> None:
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("Always-increment evaluation requires a Box action space.")
        self.action = np.ones(action_space.shape, dtype=action_space.dtype)

    def predict(self, _observation: dict[str, Any]) -> tuple[np.ndarray, None]:
        return self.action.copy(), None


# ------------ RUN SIMULATION FUNCTION ------------
def run_always_increment_policy_simulation(
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
    run_name: str = "Always_increment",
) -> float:
    """Return mean episode revenue in EUR and log evaluation results to MLflow.

    At every step the BAR level of each date in the booking window is raised by
    one (up to the table maximum), so prices only go up.
    ``eval_seed`` controls the environment and its fixed request scenario.
    """
    return run_baseline_simulation(
        lambda action_space: AlwaysIncrementPolicy(action_space),
        model="AlwaysIncrement",
        policy_name="AlwaysIncrementPolicy",
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
        bar_update_mode="always",
    )
