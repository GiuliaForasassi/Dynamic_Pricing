import csv
from io import StringIO

import pytest

from pricing_advertising.plots import occupancy_tracker as module


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


def record_episode(tracker, occupancy):
    for day_index, value in enumerate(occupancy):
        tracker.record_day(
            day_index=day_index,
            simulation_days=len(occupancy),
            occupancy_pct=value,
        )
    tracker.end_episode()


def values(metrics, key):
    return [(step, value) for name, value, step in metrics if name == key]


def test_episode_mean_daily_curve_and_csv(logged):
    metrics, texts = logged
    tracker = module.OccupancyTracker("train")

    record_episode(tracker, [0.0, 50.0, 100.0])
    tracker.log_step_metrics(step=1460)
    tracker.finalize_training()

    assert values(metrics, "occupancy_pct/by_training_timesteps/train") == [(1460, 50.0)]
    assert values(metrics, "occupancy_pct/by_stay_day/mean_completed_episodes/train") == [
        (1, 0.0),
        (2, 50.0),
        (3, 100.0),
    ]

    rows = list(csv.DictReader(StringIO(next(iter(texts.values())))))
    assert [int(row["day"]) for row in rows] == [1, 2, 3]
    assert [float(row["occupancy_pct"]) for row in rows] == [0, 50, 100]


def test_eval_rounds_are_separate_and_not_logged_twice(logged):
    metrics, _ = logged
    tracker = module.OccupancyTracker("val")

    record_episode(tracker, [0.0, 100.0])  # Media 50
    record_episode(tracker, [100.0, 100.0])  # Media 100
    tracker.log_step_metrics(step=800)
    tracker.log_step_metrics(step=800)

    record_episode(tracker, [20.0, 40.0])  # Media 30
    tracker.log_step_metrics(step=1600)

    assert values(metrics, "occupancy_pct/by_training_timesteps/val") == [(800, 75.0), (1600, 30.0)]


def test_daily_averages_exclude_partial_episode(logged):
    metrics, texts = logged
    tracker = module.OccupancyTracker("train")

    record_episode(tracker, [0.0, 100.0])
    record_episode(tracker, [100.0, 100.0])
    tracker.log_step_metrics(step=800)

    tracker.record_day(
        day_index=0,
        simulation_days=2,
        occupancy_pct=100.0,
    )
    tracker.finalize_training()
    tracker.log_step_metrics(step=1600)

    assert values(metrics, "occupancy_pct/by_stay_day/mean_completed_episodes/train") == [
        (1, 50.0),
        (2, 100.0),
    ]
    assert values(metrics, "occupancy_pct/by_training_timesteps/train") == [(800, 75.0)]
    assert len(texts) == 2


def test_only_partial_episode_produces_no_logs(logged):
    metrics, texts = logged
    tracker = module.OccupancyTracker("train")

    tracker.record_day(
        day_index=0,
        simulation_days=2,
        occupancy_pct=50.0,
    )
    tracker.log_step_metrics(step=4)
    tracker.finalize_training()

    assert not metrics
    assert not texts


def test_empty_hotel_episode_logs_zero(logged):
    metrics, _ = logged
    tracker = module.OccupancyTracker("train")

    record_episode(tracker, [0.0, 0.0])
    tracker.log_step_metrics(step=8)

    assert values(metrics, "occupancy_pct/by_training_timesteps/train") == [(8, 0.0)]


@pytest.mark.parametrize(
    "occupancy",
    [-1.0, 101.0, float("nan"), float("inf"), -float("inf")],
)
def test_invalid_percentage_is_rejected(occupancy):
    tracker = module.OccupancyTracker("train", log_to_mlflow=False)

    with pytest.raises(ValueError, match="Occupancy non valida"):
        tracker.record_day(
            day_index=0,
            simulation_days=1,
            occupancy_pct=occupancy,
        )


def test_missing_day_is_rejected():
    tracker = module.OccupancyTracker("train", log_to_mlflow=False)
    tracker.record_day(day_index=0, simulation_days=3, occupancy_pct=10.0)

    with pytest.raises(ValueError, match="fuori sequenza"):
        tracker.record_day(day_index=2, simulation_days=3, occupancy_pct=20.0)


def test_incomplete_episode_cannot_be_closed():
    tracker = module.OccupancyTracker("train", log_to_mlflow=False)
    tracker.record_day(day_index=0, simulation_days=2, occupancy_pct=10.0)

    with pytest.raises(ValueError, match="incompleto"):
        tracker.end_episode()


def test_changed_duration_is_rejected():
    tracker = module.OccupancyTracker("train", log_to_mlflow=False)
    record_episode(tracker, [10.0])

    with pytest.raises(ValueError, match="Durata cambiata"):
        tracker.record_day(day_index=0, simulation_days=2, occupancy_pct=20.0)


def test_disabled_logging_clears_pending_means(logged):
    metrics, _ = logged
    tracker = module.OccupancyTracker("val", log_to_mlflow=False)

    record_episode(tracker, [100.0])
    tracker.log_step_metrics(step=800)
    assert not metrics

    tracker.log_to_mlflow = True
    tracker.log_step_metrics(step=1600)
    assert not metrics

    record_episode(tracker, [20.0])
    tracker.log_step_metrics(step=2400)

    assert values(metrics, "occupancy_pct/by_training_timesteps/val") == [(2400, 20.0)]
