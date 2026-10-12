from collections.abc import Callable
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.plots import (
    acceptance_rate_tracker,
    occupancy_tracker,
    price_tracker,
    request_tracker,
    reward_tracker,
)
from pricing_advertising.pricing_model import bar_table
from pricing_advertising.training import (
    always_increment_policy,
    baseline_evaluation,
    constant_bar_policy,
)


@pytest.fixture
def bar_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bar_table, "BAR_TABLE", {})
    bar_table.set_bar_table_from_csv(Path(__file__).parents[1] / "data/bar_table.csv")


@pytest.fixture
def simulation(bar_tables: None, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(baseline_evaluation, "mlflow", MagicMock())
    monkeypatch.setattr(baseline_evaluation, "log_eval_artifact_plots", MagicMock())
    for module in (
        acceptance_rate_tracker,
        occupancy_tracker,
        price_tracker,
        request_tracker,
        reward_tracker,
    ):
        monkeypatch.setattr(module, "mlflow", baseline_evaluation.mlflow)
    return {
        "seed": 42,
        "eval_seed": 10_042,
        "start_simulation": date(2025, 1, 1),
        "end_simulation": date(2025, 1, 3),
        "num_category1_rooms": 100,
        "num_category2_rooms": 50,
        "num_category3_rooms": 30,
        "avg_num_requests": 5,
        "eval_episodes": 1,
        "price_plot_request_dates": [(2025, 1, 2)],
        "allow_category_alternative": False,
        "allow_period_alternative": False,
    }


def make_env(mode: str | None) -> PricingEnv:
    return PricingEnv(
        date(2025, 1, 1),
        date(2025, 1, 3),
        Hotel(100, 50, 30),
        avg_num_requests=5,
        seed=42,
        bar_update_mode=mode,  # type: ignore[arg-type]
    )


def test_invalid_bar_update_mode_is_rejected(bar_tables: None) -> None:
    with pytest.raises(ValueError, match="bar_update_mode"):
        make_env("sometimes")


def test_never_mode_keeps_bar_levels_constant(bar_tables: None) -> None:
    env = make_env("never")
    env.reset(seed=42)
    action = np.ones(env.action_space.shape, dtype=np.float32)
    for _ in range(3):
        env.step(action)
        assert np.all(env.bar_levels == 1)


def test_always_mode_increments_every_step_until_the_maximum(bar_tables: None) -> None:
    env = make_env("always")
    env.reset(seed=42)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    start, stop = env._booking_window_indices()
    env.step(action)
    assert np.all(env.bar_levels[:, start:stop] == 2)
    assert np.all(env.bar_levels[:, :start] == 1)
    env.step(action)
    assert env.bar_levels[:, start] == pytest.approx(2)
    assert np.all(env.bar_levels[:, start + 1 : stop] == 3)
    assert env.bar_levels.max() <= env._max_bar_level


def test_default_mode_keeps_policy_driven_updates(bar_tables: None) -> None:
    env = make_env(None)
    env.reset(seed=42)
    env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert np.all(env.bar_levels == 1)
    assert env.bar_update_mode is None


@pytest.mark.parametrize(
    ("run", "model", "mode"),
    [
        (constant_bar_policy.run_constant_bar_policy_simulation, "ConstantBar", "never"),
        (
            always_increment_policy.run_always_increment_policy_simulation,
            "AlwaysIncrement",
            "always",
        ),
    ],
)
def test_runner_logs_forced_mode_and_artifacts(
    simulation: dict, run: Callable[..., float], model: str, mode: str
) -> None:
    mean = run(**simulation)
    params = baseline_evaluation.mlflow.log_params.call_args.args[0]
    assert params["model"] == model
    assert params["bar_update_mode"] == mode
    assert params["training_duration"] == 0
    artifact = baseline_evaluation.log_eval_artifact_plots.call_args.args[0]
    assert len(artifact["steps"]) == 3
    assert all(step["bar_update_mode"] == mode for step in artifact["steps"])
    snapshot = next(step for step in artifact["steps"] if "snapshot_start_day" in step)
    levels = np.asarray(snapshot["bar_levels_by_category"])
    if mode == "never":
        assert np.all(levels == 1)
    else:
        assert levels.min() >= 2
    baseline_evaluation.mlflow.log_metric.assert_any_call("final_eval_mean_reward", mean)
    assert run(**simulation) == mean


def test_runs_share_the_same_request_scenario(simulation: dict) -> None:
    constant_bar_policy.run_constant_bar_policy_simulation(**simulation)
    always_increment_policy.run_always_increment_policy_simulation(**simulation)
    first, second = (
        call.args[0]["steps"] for call in baseline_evaluation.log_eval_artifact_plots.call_args_list
    )
    assert [[r["requested"] for r in step["requests"]] for step in first] == [
        [r["requested"] for r in step["requests"]] for step in second
    ]
