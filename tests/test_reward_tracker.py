import csv
from io import StringIO

import pytest

from pricing_advertising.plots import reward_tracker as module


@pytest.fixture
def logged(monkeypatch):
    metrics = []
    texts = {}

    def log_metric(key, value, step):
        metrics.append((key, value, step))

    def log_text(text, path):
        texts[path] = text

    monkeypatch.setattr(module.mlflow, "log_metric", log_metric)
    monkeypatch.setattr(module.mlflow, "log_text", log_text)
    return metrics, texts


def record_episode(tracker, rewards):
    for day_index, reward in enumerate(rewards):
        tracker.record_day(
            day_index=day_index,
            simulation_days=len(rewards),
            reward=reward,
        )
    tracker.end_episode()


def values(metrics, key):
    return [(step, value) for name, value, step in metrics if name == key]


def test_total_mean_and_csv_are_distinct(logged):
    metrics, texts = logged
    tracker = module.RewardTracker("train")

    record_episode(tracker, [100.0, 0.0, 500.0])
    tracker.log_step_metrics(step=1460)
    tracker.finalize_training()

    assert values(metrics, "episode_return/by_training_timesteps/train") == [(1460, 600.0)]
    assert values(metrics, "mean_daily_reward/by_training_timesteps/train") == [(1460, 200.0)]
    assert values(metrics, "reward/by_request_day/mean_completed_episodes/train") == [
        (1, 100.0),
        (2, 0.0),
        (3, 500.0),
    ]

    rows = list(csv.DictReader(StringIO(next(iter(texts.values())))))
    assert [int(row["day"]) for row in rows] == [1, 2, 3]
    assert [float(row["reward"]) for row in rows] == [100.0, 0.0, 500.0]


def test_eval_rounds_do_not_mix_and_do_not_log_twice(logged):
    metrics, _ = logged
    tracker = module.RewardTracker("val")

    record_episode(tracker, [100.0, 0.0])  # Totale 100
    record_episode(tracker, [300.0, 200.0])  # Totale 500
    tracker.log_step_metrics(step=800)
    tracker.log_step_metrics(step=800)  # Nessun duplicato

    record_episode(tracker, [20.0, 40.0])  # Totale 60
    tracker.log_step_metrics(step=1600)

    assert values(metrics, "episode_return/by_training_timesteps/val") == [
        (800, 300.0),
        (1600, 60.0),
    ]
    assert values(metrics, "mean_daily_reward/by_training_timesteps/val") == [
        (800, 150.0),
        (1600, 30.0),
    ]


def test_daily_averages_exclude_partial_episode(logged):
    metrics, texts = logged
    tracker = module.RewardTracker("train")

    record_episode(tracker, [100.0, 0.0])
    record_episode(tracker, [300.0, 200.0])
    tracker.log_step_metrics(step=800)

    tracker.record_day(
        day_index=0,
        simulation_days=2,
        reward=99999.0,
    )
    tracker.finalize_training()
    tracker.log_step_metrics(step=1600)

    assert values(metrics, "reward/by_request_day/mean_completed_episodes/train") == [
        (1, 200.0),
        (2, 100.0),
    ]
    assert values(metrics, "episode_return/by_training_timesteps/train") == [(800, 300.0)]
    assert len(texts) == 2


def test_zero_episode_is_logged(logged):
    metrics, _ = logged
    tracker = module.RewardTracker("train")

    record_episode(tracker, [0.0, 0.0])
    tracker.log_step_metrics(step=8)

    assert values(metrics, "episode_return/by_training_timesteps/train") == [(8, 0.0)]
    assert values(metrics, "mean_daily_reward/by_training_timesteps/train") == [(8, 0.0)]


def test_incomplete_episode_cannot_be_closed():
    tracker = module.RewardTracker("train", log_to_mlflow=False)
    tracker.record_day(day_index=0, simulation_days=2, reward=10.0)

    with pytest.raises(ValueError, match="incomplete episode"):
        tracker.end_episode()


def test_missing_day_is_rejected():
    tracker = module.RewardTracker("train", log_to_mlflow=False)
    tracker.record_day(day_index=0, simulation_days=3, reward=10.0)

    with pytest.raises(ValueError, match="out-of-sequence"):
        tracker.record_day(day_index=2, simulation_days=3, reward=20.0)


def test_changed_duration_is_rejected():
    tracker = module.RewardTracker("train", log_to_mlflow=False)
    record_episode(tracker, [10.0])

    with pytest.raises(ValueError, match="Simulation duration changed"):
        tracker.record_day(day_index=0, simulation_days=2, reward=20.0)


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_reward_is_rejected(reward):
    tracker = module.RewardTracker("train", log_to_mlflow=False)

    with pytest.raises(ValueError, match="Reward must be finite"):
        tracker.record_day(day_index=0, simulation_days=1, reward=reward)


def test_disabled_logging_clears_pending_totals(logged):
    metrics, _ = logged
    tracker = module.RewardTracker("val", log_to_mlflow=False)

    record_episode(tracker, [100.0])
    tracker.log_step_metrics(step=800)
    assert not metrics

    tracker.log_to_mlflow = True
    tracker.log_step_metrics(step=1600)
    assert not metrics

    record_episode(tracker, [20.0])
    tracker.log_step_metrics(step=2400)

    assert values(metrics, "episode_return/by_training_timesteps/val") == [(2400, 20.0)]


def test_only_partial_episode_produces_no_logs(logged):
    metrics, texts = logged
    tracker = module.RewardTracker("train")

    tracker.record_day(day_index=0, simulation_days=2, reward=100.0)
    tracker.log_step_metrics(step=4)
    tracker.finalize_training()

    assert not metrics
    assert not texts
