import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import optuna
import pytest

from pricing_advertising.pricing_model import bar_table
from pricing_advertising.training import optuna_tuning as tuning


@pytest.fixture
def bar_csv(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(bar_table, "BAR_TABLE", {})
    path = Path(__file__).parents[1] / "data/bar_table.csv"
    bar_table.set_bar_table_from_csv(path)
    return path


@pytest.mark.parametrize(
    "changes",
    [
        {"timesteps": 500},
        {"n_envs": 0},
        {"scenario_trials": 0},
        {"ppo_trials": -1},
        {"target_occupancy": 1.0},
        {"end": date(2024, 1, 1)},
        {"train_seeds": ()},
        {"train_seeds": (42, 43)},
        {"validation_seeds": tuning.Settings().holdout_seeds},
        {"holdout_seeds": (tuning.SEEDS[0] + 1,)},
        {"sampler_seed": -1},
        {"train_seeds": (True,)},
        {"validation_seeds": (42.5,)},
    ],
)
def test_invalid_settings(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must"):
        replace(tuning.Settings(), **changes).validate()


def test_occupancy_objective_does_not_cancel_extremes() -> None:
    results = [{"occupancy": [0.4, 0.7, 1.0]}, {"occupancy": [1.0, 0.7, 0.4]}]
    assert tuning.occupancy_loss(results, 0.7) == pytest.approx(0.2)
    assert tuning.occupancy_loss([{"occupancy": [0.7] * 3}], 0.7) == 0


@pytest.mark.parametrize("n_envs", [1, 4])
def test_search_space_has_equal_budgets_and_divisible_batches(n_envs: int) -> None:
    settings = replace(tuning.Settings(), n_envs=n_envs)
    settings.validate()
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
    for _ in range(30):
        trial = study.ask()
        params = tuning.sample_ppo(trial, smoke=False)
        rollout = params["n_steps"] * n_envs
        assert settings.timesteps % rollout == 0
        assert rollout % params["batch_size"] == 0
        assert params["batch_size"] <= rollout
        study.tell(trial, 0.0)


def test_evaluation_is_reproducible_and_closes_env(
    bar_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert bar_csv.exists()
    settings = replace(tuning.Settings(), end=date(2025, 1, 3))
    scenario = tuning.Scenario(100, 50, 30, 5)
    original = tuning.make_env
    envs = []

    def make_env(
        scenario: tuning.Scenario, settings: tuning.Settings, seed: int
    ) -> tuning.PricingEnv:
        env = original(scenario, settings, seed)
        monkeypatch.setattr(env, "close", MagicMock())
        envs.append(env)
        return env

    monkeypatch.setattr(tuning, "make_env", make_env)
    first = tuning.evaluate(scenario, settings, (42, 43))
    assert first == tuning.evaluate(scenario, settings, (42, 43))
    for result in first:
        assert result["revenue"] >= 0
        assert len(result["occupancy"]) == 3
        assert np.all(np.asarray(result["occupancy"]) >= 0)
        assert np.all(np.asarray(result["occupancy"]) <= 1)
    model = MagicMock()
    model.predict.side_effect = RuntimeError("prediction failed")
    with pytest.raises(RuntimeError, match="prediction failed"):
        tuning.evaluate(scenario, settings, (42,), model)
    for env in envs:
        assert isinstance(env.close, MagicMock)
        env.close.assert_called_once()


def test_two_stages_freeze_scenario_and_keep_holdout_separate(
    bar_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = replace(tuning.Settings(), scenario_trials=2, ppo_trials=2)
    events: list[tuple[str, int | tuple[int, ...]]] = []
    trained_scenarios = []

    def evaluate(
        scenario: tuning.Scenario,
        settings: tuning.Settings,
        seeds: tuple[int, ...],
        model: tuning.PPO | None = None,
    ) -> list[dict[str, Any]]:
        events.append(("evaluate", seeds))
        occupancy = (
            [settings.target_occupancy] * 3 if scenario.rooms == (100, 50, 30) else [0.1] * 3
        )
        return [
            {"seed": seed, "revenue": 120.0 if model is not None else 100.0, "occupancy": occupancy}
            for seed in seeds
        ]

    def train_model(
        scenario: tuning.Scenario,
        _settings: tuning.Settings,
        _parameters: dict[str, Any],
        seed: int,
        _output: Path,
    ) -> MagicMock:
        events.append(("train", seed))
        trained_scenarios.append(scenario)
        return MagicMock()

    monkeypatch.setattr(tuning, "evaluate", evaluate)
    monkeypatch.setattr(tuning, "train_model", train_model)
    monkeypatch.setattr(tuning.PPO, "load", MagicMock())
    output = tmp_path / "study"
    report = tuning.run_study(settings, output, bar_csv)
    assert report["scenario"] == {
        "num_category1_rooms": 100,
        "num_category2_rooms": 50,
        "num_category3_rooms": 30,
        "avg_num_requests": 60,
    }
    assert report["scenario_occupancy_mae"] == 0
    assert report["validation_mean_revenue"] == 120
    assert report["holdout_mean_paired_delta"] == 20
    assert len(trained_scenarios) == 6
    assert all(scenario == tuning.Scenario(100, 50, 30, 60) for scenario in trained_scenarios)
    first_training = next(i for i, event in enumerate(events) if event[0] == "train")
    last_training = max(i for i, event in enumerate(events) if event[0] == "train")
    assert all(
        i < first_training
        for i, event in enumerate(events)
        if event == ("evaluate", settings.calibration_seeds)
    )
    assert all(
        i > last_training
        for i, event in enumerate(events)
        if event == ("evaluate", settings.holdout_seeds)
    )
    saved = json.loads((output / "results.json").read_text())
    assert saved["ppo_parameters"] == report["ppo_parameters"]
    assert saved["holdout_mean_paired_delta"] == 20
    study = optuna.load_study(study_name="ppo", storage=f"sqlite:///{output / 'studies.sqlite3'}")
    assert len(study.trials) == 2
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    with pytest.raises(FileExistsError):
        tuning.run_study(settings, output, bar_csv)
