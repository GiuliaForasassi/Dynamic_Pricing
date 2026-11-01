import csv
from io import StringIO
from unittest.mock import Mock

import pytest

from pricing_advertising.plots import cancellation_tracker as module


@pytest.fixture
def logged(monkeypatch):
    logging = Mock()
    monkeypatch.setattr(module, "mlflow", logging)
    return logging


def record_episode(tracker, counts):
    for day_index, count in enumerate(counts):
        tracker.record_day(day_index=day_index, simulation_days=len(counts), num_cancelled=count)
    tracker.end_episode()


def test_counts_are_averaged_by_episode_not_by_reservations(logged):
    tracker = module.CancellationTracker("val")
    record_episode(tracker, [0, 120, 4])
    record_episode(tracker, [0, 100, 8])
    tracker.log_step_metrics(1000)
    tracker.log_step_metrics(1000)
    tracker.record_day(day_index=0, simulation_days=3, num_cancelled=999)
    tracker.finalize_training()
    daily = [
        (call.kwargs["step"], call.args[1])
        for call in logged.log_metric.call_args_list
        if call.args[0] == "cancellation_amount/by_cancellation_day/mean_completed_episodes/val"
    ]
    assert daily == [(1, 0), (2, 110), (3, 6)]
    assert logged.log_metric.call_count == 4
    assert logged.log_text.call_count == 2
    text, path = logged.log_text.call_args_list[0].args
    assert path == "episodes/val/episode_00000_cancellation.csv"
    assert [int(row["cancellation_amount"]) for row in csv.DictReader(StringIO(text))] == [
        0,
        120,
        4,
    ]


@pytest.mark.parametrize("count", [-1, 0.5, float("nan"), float("inf")])
def test_invalid_count_is_rejected(count):
    tracker = module.CancellationTracker("train", log_to_mlflow=False)
    with pytest.raises(ValueError, match="Cancellation count"):
        tracker.record_day(day_index=0, simulation_days=1, num_cancelled=count)


def test_sequence_and_completion_validation():
    tracker = module.CancellationTracker("train", log_to_mlflow=False)
    with pytest.raises(ValueError, match="positive"):
        tracker.record_day(day_index=0, simulation_days=0, num_cancelled=0)
    tracker.record_day(day_index=0, simulation_days=2, num_cancelled=0)
    with pytest.raises(ValueError, match="out-of-sequence"):
        tracker.record_day(day_index=0, simulation_days=2, num_cancelled=0)
    with pytest.raises(ValueError, match="Duration changed"):
        tracker.record_day(day_index=1, simulation_days=3, num_cancelled=0)
    with pytest.raises(ValueError, match="incomplete"):
        tracker.end_episode()


@pytest.mark.parametrize("enabled", [False, True])
def test_partial_episode_does_not_log(logged, enabled):
    tracker = module.CancellationTracker("train", log_to_mlflow=enabled)
    tracker.end_episode()
    tracker.record_day(day_index=0, simulation_days=2, num_cancelled=4)
    tracker.finalize_training()
    tracker.log_step_metrics(1)
    assert not logged.mock_calls


def test_disabled_logging_and_zero_counts(logged):
    tracker = module.CancellationTracker("train", log_to_mlflow=False)
    record_episode(tracker, [0, 0])
    tracker.log_step_metrics(1)
    tracker.finalize_training()
    assert not logged.mock_calls
