from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest

from pricing_advertising.plots import (
    acceptance_rate_tracker,
    occupancy_tracker,
    price_tracker,
    request_tracker,
    reward_tracker,
)
from pricing_advertising.pricing_model import bar_table
from pricing_advertising.training import random_policy


@pytest.fixture
def simulation(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(bar_table, "BAR_TABLE", {})
    bar_table.set_bar_table_from_csv(Path(__file__).parents[1] / "data/bar_table.csv")
    monkeypatch.setattr(random_policy, "mlflow", MagicMock())
    monkeypatch.setattr(random_policy, "log_eval_artifact_plots", MagicMock())
    for module in (
        acceptance_rate_tracker,
        occupancy_tracker,
        price_tracker,
        request_tracker,
        reward_tracker,
    ):
        monkeypatch.setattr(module, "mlflow", random_policy.mlflow)
    return {
        "seed": 42,
        "eval_seed": 10_042,
        "start_simulation": date(2025, 1, 1),
        "end_simulation": date(2025, 1, 3),
        "num_category1_rooms": 100,
        "num_category2_rooms": 50,
        "num_category3_rooms": 30,
        "avg_num_requests": 5,
        "eval_episodes": 2,
        "price_plot_request_dates": [(2025, 1, 2)],
        "allow_category_alternative": False,
        "allow_period_alternative": False,
    }


def test_random_policy_samples_reproducible_uniform_actions() -> None:
    def make_policy(seed: int) -> random_policy.RandomPolicy:
        return random_policy.RandomPolicy(
            gym.spaces.Box(0, 1, shape=(3, 2, 4), dtype=np.float32), seed
        )

    policy = make_policy(42)
    same_seed = make_policy(42)
    other_seed = make_policy(43)
    actions = []
    for _ in range(1000):
        action, state = policy.predict({})
        assert state is None
        assert policy.action_space.contains(action)
        np.testing.assert_array_equal(action, same_seed.predict({"ignored": True})[0])
        actions.append(action)
    assert not np.array_equal(actions[0], other_seed.predict({})[0])
    assert not np.array_equal(actions[0], actions[1])
    samples = np.asarray(actions)
    assert abs(float(samples.mean()) - 0.5) < 0.01
    assert abs(float(samples.var()) - 1 / 12) < 0.005


def test_simulation_logs_raw_rewards_and_complete_episodes(simulation: dict) -> None:
    mean = random_policy.run_random_policy_simulation(**simulation)
    random_policy.mlflow.start_run.assert_called_once_with(run_name="Random_uniform")
    artifacts = [call.args[0] for call in random_policy.log_eval_artifact_plots.call_args_list]
    assert len(artifacts) == 2
    rewards = []
    for episode_id, artifact in enumerate(artifacts):
        assert artifact["episode_id"] == episode_id
        assert artifact["checkpoint_timesteps"] == 0
        assert artifact["schema_version"] == 3
        assert artifact["snapshot_dates"] == ["2025-01-02"]
        assert [step["request_day"] for step in artifact["steps"]] == [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
        ]
        rewards.append(sum(step["step_reward"] for step in artifact["steps"]))
    assert mean == np.mean(rewards)
    random_policy.mlflow.log_metric.assert_any_call("final_eval_mean_reward", mean)
    random_policy.mlflow.log_metric.assert_any_call("final_eval_std_reward", float(np.std(rewards)))
    random_policy.mlflow.log_dict.assert_any_call(
        {
            "seed": 42,
            "eval_request_seed": 10_042,
            "episode_rewards": rewards,
            "episode_lengths": [3, 3],
        },
        "eval_data/results.json",
    )
    params = random_policy.mlflow.log_params.call_args.args[0]
    assert params["training_duration"] == 0
    assert params["eval_norm_reward"] is False
    assert params["allow_category_alternative"] is False
    assert params["allow_period_alternative"] is False
    assert random_policy.run_random_policy_simulation(**simulation) == mean
    assert random_policy.log_eval_artifact_plots.call_args_list[2].args[0] == artifacts[0]


def test_action_seed_does_not_change_request_scenario(simulation: dict) -> None:
    random_policy.run_random_policy_simulation(**simulation)
    first = random_policy.log_eval_artifact_plots.call_args_list[0].args[0]["steps"]
    random_policy.run_random_policy_simulation(**{**simulation, "seed": 43})
    second = random_policy.log_eval_artifact_plots.call_args_list[2].args[0]["steps"]
    assert first[0]["policy_coefficients"] != second[0]["policy_coefficients"]
    assert [[r["requested"] for r in step["requests"]] for step in first] == [
        [r["requested"] for r in step["requests"]] for step in second
    ]


def test_simulation_keeps_requests_and_advances_rngs(
    simulation: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset = random_policy.PricingEnv.reset
    seeds = []

    def record_reset(
        self: random_policy.PricingEnv,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        seeds.append(seed)
        return reset(self, seed=seed, options=options)

    monkeypatch.setattr(random_policy.PricingEnv, "reset", record_reset)
    random_policy.run_random_policy_simulation(**simulation)
    assert seeds == [10_042, None]
    episodes = [
        call.args[0]["steps"] for call in random_policy.log_eval_artifact_plots.call_args_list
    ]
    assert episodes[0][0]["policy_coefficients"] != episodes[1][0]["policy_coefficients"]
    assert [[r["requested"] for r in step["requests"]] for step in episodes[0]] == [
        [r["requested"] for r in step["requests"]] for step in episodes[1]
    ]


def test_simulation_uses_existing_plot_helpers(
    simulation: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pricing_advertising.plots import eval_artifact_plots

    monkeypatch.setattr(eval_artifact_plots, "mlflow", random_policy.mlflow)
    monkeypatch.setattr(
        random_policy, "log_eval_artifact_plots", eval_artifact_plots.log_eval_artifact_plots
    )
    random_policy.run_random_policy_simulation(**{**simulation, "eval_episodes": 1})
    assert random_policy.mlflow.log_figure.call_count > 0


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("eval_episodes", 0),
        ("eval_episodes", 1.5),
        ("eval_episodes", True),
        ("seed", -1),
        ("seed", True),
        ("eval_seed", 2**32),
        ("avg_num_requests", 0),
        ("avg_num_requests", 1.5),
        ("end_simulation", date(2024, 12, 31)),
        ("price_plot_request_dates", [(2025, 1, 4)]),
    ],
)
def test_invalid_configuration_fails_before_logging(
    simulation: dict, parameter: str, value: object
) -> None:
    with pytest.raises(ValueError, match="must"):
        random_policy.run_random_policy_simulation(**{**simulation, parameter: value})
    random_policy.mlflow.start_run.assert_not_called()


def test_simulation_closes_environment_on_error(
    simulation: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    close = MagicMock()
    monkeypatch.setattr(random_policy.PricingEnv, "close", close)
    random_policy.log_eval_artifact_plots.side_effect = RuntimeError("Plot failed")
    with pytest.raises(RuntimeError, match="Plot failed"):
        random_policy.run_random_policy_simulation(**simulation)
    close.assert_called_once()


def test_custom_run_name_preserves_rewards(simulation: dict) -> None:
    default_reward = random_policy.run_random_policy_simulation(**simulation)
    random_policy.mlflow.start_run.reset_mock()
    custom_reward = random_policy.run_random_policy_simulation(
        **simulation, run_name="Random_baseline_60_requests"
    )
    random_policy.mlflow.start_run.assert_called_once_with(run_name="Random_baseline_60_requests")
    assert custom_reward == default_reward


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


@pytest.mark.parametrize("episodes", [1, 2])
def test_tracker_metrics_match_episode_data(
    simulation: dict, monkeypatch: pytest.MonkeyPatch, episodes: int
) -> None:
    step = random_policy.PricingEnv.step
    recorded = []

    def record_step(
        self: random_policy.PricingEnv, action: np.ndarray
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        result = step(self, action)
        recorded.append(deepcopy(result[4]["step_metrics"]))
        return result

    monkeypatch.setattr(random_policy.PricingEnv, "step", record_step)
    mean = random_policy.run_random_policy_simulation(
        **{**simulation, "eval_episodes": episodes, "price_plot_request_dates": []}
    )
    days = 3
    assert len(recorded) == days * episodes
    actual = {}
    for call in random_policy.mlflow.log_metric.call_args_list:
        key, value = call.args
        actual.setdefault(key, []).append((call.kwargs.get("step", 0), value))
    assert not any(key.endswith("/train") for key in actual)

    requests = np.zeros(days)
    offers = np.zeros(days)
    accepted = np.zeros(days)
    confirmed = np.zeros(days)
    stays = np.zeros(days)
    occupancy = np.zeros(days)
    rewards = np.zeros(days)
    booked_revenue = np.zeros(days)
    booked_nights = np.zeros(days)
    for metrics in recorded:
        day = metrics["day_index"]
        requests[day] += metrics["num_requests"]
        offers[day] += metrics["num_offers"]
        accepted[day] += metrics["num_price_accepted"]
        confirmed[day] += metrics["num_confirmed"]
        occupancy[day] += metrics["occupancy_pct"]
        rewards[day] += metrics["reward"]
        for check_in, nights in metrics["request_stays"]:
            stays[check_in : min(check_in + nights, days)] += 1
        prices = iter(metrics["confirmed_nightly_prices"])
        for request in metrics["confirmed_requests"]:
            check_in = (request.t_reservation - simulation["start_simulation"]).days
            for night in range(request.days_stay):
                price = next(prices)
                stay_day = check_in + night
                if 0 <= stay_day < days:
                    booked_revenue[stay_day] += price
                    booked_nights[stay_day] += 1

    summary = {
        "occupancy_pct": occupancy.sum() / (days * episodes),
        "num_requests_per_episode": requests.sum() / episodes,
        "episode_return": mean,
        "mean_daily_reward": mean / days,
        "booked_mean_price": _ratio(booked_revenue.sum(), booked_nights.sum()),
        "confirmation_rate": _ratio(confirmed.sum(), requests.sum()),
        "price_acceptance_rate": _ratio(accepted.sum(), offers.sum()),
    }
    for name, value in summary.items():
        np.testing.assert_allclose(
            actual[f"{name}/by_training_timesteps/val"], [(0, value)], equal_nan=True
        )
    daily = {
        "occupancy_pct/by_stay_day/mean_completed_episodes": occupancy / episodes,
        "reward/by_request_day/mean_completed_episodes": rewards / episodes,
        "num_requests/by_request_day/mean_completed_episodes": requests / episodes,
        "num_requests/by_stay_day/mean_completed_episodes": stays / episodes,
        "booked_mean_price/by_stay_day/pooled_completed_episodes": [
            _ratio(revenue, nights)
            for revenue, nights in zip(booked_revenue, booked_nights, strict=True)
        ],
        "confirmation_rate/by_request_day_pooled_completed_episodes": [
            _ratio(count, total) for count, total in zip(confirmed, requests, strict=True)
        ],
        "price_acceptance_rate/by_request_day_pooled_completed_episodes": [
            _ratio(count, total) for count, total in zip(accepted, offers, strict=True)
        ],
    }
    for name, values in daily.items():
        np.testing.assert_allclose(
            actual[f"{name}/val"], list(enumerate(values, start=1)), equal_nan=True
        )
    csv_paths = {call.args[1] for call in random_policy.mlflow.log_text.call_args_list}
    assert csv_paths == {
        f"episodes/val/episode_{episode:05d}_{suffix}.csv"
        for episode in range(episodes)
        for suffix in ("occupancy", "request_counts", "reward", "price")
    }


def test_trackers_do_not_change_random_trajectory(
    simulation: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracked_reward = random_policy.run_random_policy_simulation(**simulation)
    tracked_artifacts = random_policy.log_eval_artifact_plots.call_args_list.copy()
    random_policy.log_eval_artifact_plots.reset_mock()
    monkeypatch.setattr(random_policy, "EpisodeMetrics", MagicMock())
    assert random_policy.run_random_policy_simulation(**simulation) == tracked_reward
    assert random_policy.log_eval_artifact_plots.call_args_list == tracked_artifacts
