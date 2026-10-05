import csv
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO

from pricing_advertising.diagnostics import fixed_policies as diagnostics
from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.pricing_model import bar_table


@pytest.fixture
def short_simulation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "START", date(2025, 1, 1))
    monkeypatch.setattr(diagnostics, "END", date(2025, 1, 3))
    monkeypatch.setattr(diagnostics, "AVG_NUM_REQUESTS", 5)
    monkeypatch.setattr(diagnostics, "SNAPSHOT_DAYS", (0, 2, 30))
    monkeypatch.setattr(bar_table, "BAR_TABLE", {})
    bar_table.set_bar_table_from_csv(Path(__file__).parents[1] / "data" / "bar_table.csv")


def make_env() -> PricingEnv:
    return PricingEnv(
        diagnostics.START,
        diagnostics.END,
        Hotel(*diagnostics.ROOMS),
        avg_num_requests=diagnostics.AVG_NUM_REQUESTS,
        seed=42,
    )


@pytest.mark.usefixtures("short_simulation")
def test_episode_preserves_dynamics_and_exports(tmp_path: Path) -> None:
    policy = diagnostics.make_policies(3, 42)["theta=0 (sale piano)"]
    result = diagnostics.run_episode(policy, 42, 0.995)
    env = make_env()
    observation, _ = env.reset(seed=42)
    revenue = 0.0
    for day, row in enumerate(result["daily_diagnostics"]):
        np.testing.assert_allclose(
            np.clip(result["demand_before_action"][day], 0, 1),
            observation["demand"],
            atol=1e-7,
        )
        levels_before = env.bar_levels.copy()
        observation, reward, _, _, info = env.step(policy(observation))
        assert reward == row["reward"]
        revenue += reward
        expected_before, expected_after = [], []
        for offer in env.last_valid_requests:
            category = offer.category.value - 1
            start = (offer.t_reservation - env.start_simulation).days
            for night in range(start, start + offer.days_stay):
                if night > day:
                    expected_before.append(levels_before[category, night])
                    expected_after.append(env.bar_levels[category, night])
        assert row["offered_future_nights"] == len(expected_before)
        assert row["offered_at_max_before"] == sum(
            level == env.n_levels - 1 for level in expected_before
        )
        assert row["offered_changed"] == sum(
            after > before for before, after in zip(expected_before, expected_after, strict=True)
        )
        assert row["sold_nights"] == len(info["step_metrics"]["confirmed_nightly_prices"])
    env.close()
    assert result["revenue"] == revenue
    diagnostics.write_episode_diagnostics(result, tmp_path, "fixed")
    with (tmp_path / "daily_fixed_42.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 3
    with (tmp_path / "demand_fixed_42.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 * 3 * diagnostics.NUM_WINDOWS
    assert {row["category"] for row in rows} == {"1", "2", "3"}


@pytest.mark.usefixtures("short_simulation")
def test_raw_demand_detects_clipping_before_action() -> None:
    env = make_env()
    env.demand[0, 1 : 1 + diagnostics.WINDOW_SIZE] = 2 * env.avg_num_requests
    raw = diagnostics.raw_demand_windows(env)
    assert raw[0, 0] == 2
    assert env.get_observation()["demand"][0, 0] == 1
    assert np.count_nonzero(raw > 1) == 1
    env.close()


def test_summary_uses_night_weighted_ratios_and_raw_rewards() -> None:
    result = {
        "daily_diagnostics": [
            {
                "offered_future_nights": 1,
                "offered_at_max_before": 1,
                "offered_changed": 0,
                "sold_nights": 1,
                "sold_at_zero": 1,
                "reward": 0,
                "arrival_at_max_fraction": 0,
            },
            {
                "offered_future_nights": 9,
                "offered_at_max_before": 0,
                "offered_changed": 3,
                "sold_nights": 9,
                "sold_at_zero": 0,
                "reward": 1000,
                "arrival_at_max_fraction": 1,
            },
        ],
        "demand_before_action": np.asarray([0, 1, 2, 0]),
    }
    summary = diagnostics.summarize_diagnostics(result)
    assert summary["offered_at_max_before_fraction"] == 0.1
    assert summary["offered_changed_fraction"] == 0.3
    assert summary["sold_at_zero_fraction"] == 0.1
    assert summary["demand_clipped_fraction"] == 0.25
    assert summary["reward_zero_fraction"] == 0.5
    assert summary["reward_p50_eur"] == 500
    assert summary["reward_std_eur"] == 500
    assert not any("scaled" in key or "divisor" in key for key in summary)


def test_fixed_policies_emit_six_curves():
    for policy in diagnostics.make_policies(3, 42).values():
        actions = policy({})
        assert actions.shape == (3, 2, 4)
        assert np.all((actions >= 0) & (actions <= 1))


@pytest.mark.usefixtures("short_simulation")
def test_diagnostics_rejects_old_checkpoint(monkeypatch, tmp_path):
    env = make_env()
    env.action_space = diagnostics.gym.spaces.Box(0, 1, shape=(3, 4), dtype=np.float32)
    model = PPO("MultiInputPolicy", env, n_steps=2, batch_size=2, device="cpu")
    model_path = tmp_path / "old_model.zip"
    model.save(model_path)
    env.close()
    monkeypatch.setattr(diagnostics, "MODEL_PATH", model_path)
    with pytest.raises(ValueError, match="Train a new PPO model"):
        diagnostics.main()


@pytest.mark.usefixtures("short_simulation")
def test_no_offers_produces_undefined_ratios(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "AVG_NUM_REQUESTS", 0)
    policy = diagnostics.make_policies(3, 42)["theta=0 (sale piano)"]
    result = diagnostics.run_episode(policy, 42, 0.995)
    summary = diagnostics.summarize_diagnostics(result)
    assert result["revenue"] == 0
    assert summary["reward_zero_fraction"] == 1
    assert np.isnan(summary["offered_at_max_before_fraction"])
    assert np.isnan(summary["offered_changed_fraction"])
    assert np.isnan(summary["sold_at_zero_fraction"])


def test_paired_differences_match_seed_not_position() -> None:
    runs = [{"seed": 2, "revenue": 25}, {"seed": 1, "revenue": 8}]
    baseline = [{"seed": 1, "revenue": 10}, {"seed": 2, "revenue": 20}]
    np.testing.assert_array_equal(
        diagnostics.paired_differences(runs, baseline, "revenue"), [5, -2]
    )
    with pytest.raises(ValueError, match="stessi seed"):
        diagnostics.paired_differences(runs[:1], baseline, "revenue")
    with pytest.raises(ValueError, match="duplicati"):
        diagnostics.paired_differences([runs[0], runs[0]], baseline, "revenue")


@pytest.mark.usefixtures("short_simulation")
def test_main_writes_all_policy_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = make_env()
    model = PPO("MultiInputPolicy", env, n_steps=2, batch_size=2, seed=42, device="cpu")
    model_path = tmp_path / "model.zip"
    model.save(model_path)
    env.close()
    monkeypatch.setattr(diagnostics, "MODEL_PATH", model_path)
    monkeypatch.setattr(diagnostics, "RESULTS_PATH", tmp_path / "policy_comparison.csv")
    monkeypatch.setattr(diagnostics, "N_SEEDS", 1)
    diagnostics.main()
    with (tmp_path / "policy_comparison.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert "offered_at_max_before_fraction" in rows[0]
    assert len(list(tmp_path.glob("daily_*.csv"))) == 6
    assert len(list(tmp_path.glob("demand_*.csv"))) == 6
    assert len(list(tmp_path.glob("actions_*.npz"))) == 2
    with (tmp_path / "paired_differences.csv").open() as handle:
        paired = list(csv.DictReader(handle))
    assert len(paired) == 10
    assert {row["metric"] for row in paired} == {"revenue", "discounted_revenue"}
    output = capsys.readouterr().out
    assert "raw < 0" in output
    assert "std delta=nan" in output
    assert "Probabilita di incremento" in output
