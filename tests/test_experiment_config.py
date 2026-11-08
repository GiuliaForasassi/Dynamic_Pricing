import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from pricing_advertising.config import ExperimentConfig, load_config


@pytest.fixture
def values():
    return {
        "run": {"run_name": "test", "seeds": [10, 20], "bar_table_path": "../bar.csv"},
        "simulation": {
            "start_simulation": "2025-01-01",
            "end_simulation": "2025-01-28",
            "num_category1_rooms": 0,
            "num_category2_rooms": 5,
            "num_category3_rooms": 3,
            "avg_num_requests": 10,
        },
        "ppo": {
            "total_timesteps": 129,
            "n_envs": 2,
            "n_steps": 32,
            "learning_rate": 0.0003,
            "gamma": 0.999,
            "n_epochs": 1,
            "batch_size": 32,
            "gae_lambda": 0.98,
            "clip_range": 0.2,
            "net_arch": {"pi": [64], "vf": [64]},
        },
        "evaluation": {
            "eval_frequency": 32,
            "eval_seeds": [100],
            "price_plot_request_dates": ["2025-01-07"],
        },
        "cancellation": {},
    }


def test_load_and_adapt_config(values, tmp_path):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    config = load_config(path)
    assert config.run.bar_table_path == (tmp_path / "../bar.csv").resolve()
    assert config.summary()["effective_timesteps"] == 192
    assert config.summary()["scheduled_evaluations"] == 3
    kwargs = config.runner_kwargs()
    assert kwargs["start_simulation"] == date(2025, 1, 1)
    assert kwargs["price_plot_request_dates"] == [(2025, 1, 7)]
    assert kwargs["cancellation_config"]["daily_probability"] == 0.005
    assert kwargs["cancellation_config"]["timing_strength"] == 0.3
    assert kwargs["cancellation_config"]["timing_decay_days"] == 3.0
    assert kwargs["eval_seeds"] == [100]


@pytest.mark.parametrize("strength", [0.0, 0.6])
def test_custom_cancellation_timing_is_forwarded(values, strength):
    values["cancellation"] = {"timing_strength": strength, "timing_decay_days": 5.0}
    config = ExperimentConfig.model_validate(values)
    cancellation = config.runner_kwargs()["cancellation_config"]
    assert cancellation["timing_strength"] == strength
    assert cancellation["timing_decay_days"] == 5.0


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("ppo", "n_envs", True),
        ("ppo", "n_steps", 0),
        ("ppo", "net_arch", {"pi": [64], "wrong": [64]}),
        ("simulation", "num_category1_rooms", -1),
        ("simulation", "end_simulation", "2024-12-31"),
        ("evaluation", "eval_seeds", [21]),
        ("evaluation", "price_plot_request_dates", ["2025-02-01"]),
        ("cancellation", "daily_probability", 2),
        ("cancellation", "timing_strength", -1),
        ("cancellation", "timing_strength", float("inf")),
        ("cancellation", "timing_strength", float("nan")),
        ("cancellation", "timing_decay_days", 0),
        ("cancellation", "timing_decay_days", -1),
        ("cancellation", "timing_decay_days", float("inf")),
        ("cancellation", "timing_decay_days", float("nan")),
        ("run", "unexpected", 1),
    ],
)
def test_invalid_config(values, section, key, value):
    values[section][key] = value
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(values)


def test_cancellation_disabled_and_schedule_warning(values):
    values["cancellation"] = None
    values["evaluation"]["eval_frequency"] = 1000
    config = ExperimentConfig.model_validate(values)
    assert config.runner_kwargs()["cancellation_config"] is None
    assert config.warnings() == ["no evaluation checkpoint is scheduled during training"]


def test_all_zero_room_counts_rejected(values):
    values["simulation"].update(num_category1_rooms=0, num_category2_rooms=0, num_category3_rooms=0)
    with pytest.raises(ValidationError, match="at least one room"):
        ExperimentConfig.model_validate(values)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("ppo", "gamma", True),
        ("simulation", "end_simulation", "9999-12-31"),
        ("run", "seeds", [2**32 - 1]),
        ("run", "seeds", [10, 10]),
    ],
)
def test_additional_invalid_config(values, section, key, value):
    values[section][key] = value
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(values)


def test_default_cli_passes_quick_config_and_tracking_override(monkeypatch, capsys):
    from pricing_advertising.__main__ import main
    from pricing_advertising.pricing_model import bar_table
    from pricing_advertising.training import multi_seed_training

    calls = []
    monkeypatch.setattr(bar_table, "set_bar_table_from_csv", lambda path: None)
    monkeypatch.setattr(
        multi_seed_training,
        "run_multi_seed_simulation",
        lambda **kwargs: calls.append(kwargs) or 1.0,
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///override.db")
    main([])
    assert calls[0]["run_name"] == "PPO_quick"
    assert calls[0]["resolved_config"]["run"]["tracking_uri"] == "sqlite:///override.db"
    assert calls[0]["resolved_config"]["run"]["bar_table_path"].endswith("data/bar_table.csv")
    assert "mean over 1 seeds" in capsys.readouterr().out


def test_check_config_and_invalid_config_do_not_run(values, monkeypatch, tmp_path, capsys):
    from pricing_advertising.__main__ import main
    from pricing_advertising.training import multi_seed_training

    def unexpected(**kwargs):
        pytest.fail("training must not start")

    monkeypatch.setattr(multi_seed_training, "run_multi_seed_simulation", unexpected)
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(values))
    main(["--config", str(path), "--check-config"])
    assert "scheduled_evaluations = 3" in capsys.readouterr().out
    path.write_text("{invalid")
    with pytest.raises(SystemExit) as error:
        main(["--config", str(path)])
    assert error.value.code == 2
    assert str(path) in capsys.readouterr().err


def test_presets_preserve_long_and_schedule_quick():
    root = Path(__file__).resolve().parents[1]
    long = load_config(root / "configs/long.json")
    quick = load_config(root / "configs/quick.json")
    from pricing_advertising.utils.constants import EVAL_SEEDS, SEEDS

    assert long.run.seeds == SEEDS
    assert long.evaluation.eval_seeds == EVAL_SEEDS
    assert long.ppo.total_timesteps == 1_000_000
    assert long.summary()["rollout_size"] == 2920
    assert long.runner_kwargs()["price_plot_request_dates"] == [
        (2025, 2, 28),
        (2025, 5, 15),
        (2025, 8, 1),
        (2025, 11, 12),
    ]
    assert quick.summary()["scheduled_evaluations"] == 2
    assert quick.summary()["expected_logged_episodes_per_env"] > 1
    assert quick.run.bar_table_path.is_file()
