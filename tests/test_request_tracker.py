import csv
from io import StringIO

import pytest

from pricing_advertising.plots import request_tracker as module


@pytest.fixture
def logged(monkeypatch):
    metrics = {}
    texts = {}

    def log_metric(key, value, step):
        metrics[(key, step)] = value

    def log_text(text, path):
        texts[path] = text

    monkeypatch.setattr(module.mlflow, "log_metric", log_metric)
    monkeypatch.setattr(module.mlflow, "log_text", log_text)
    return metrics, texts


def record(tracker, day, days, stays):
    tracker.record_day(
        day_index=day,
        simulation_days=days,
        num_requests=len(stays),
        request_stays=stays,
    )


def curve(metrics, axis, days):
    key = f"num_requests/{axis}/mean_completed_episodes/train"
    return [metrics[(key, day)] for day in range(1, days + 1)]


def test_arrivals_and_stay_nights_are_different(logged):
    metrics, texts = logged
    tracker = module.RequestTracker("train")

    # Indici 0-based:
    # A: notti 1, 2.
    # B: notti 2, 3, 4; solo la 2 è dentro il grafico di tre giorni.
    # C: soggiorno completamente fuori dal grafico.
    record(tracker, 0, 3, [(1, 2), (2, 3), (5, 1)])
    record(tracker, 1, 3, [])
    record(tracker, 2, 3, [])

    tracker.end_episode()
    tracker.finalize_training()

    assert curve(metrics, "by_request_day", 3) == [3, 0, 0]
    assert curve(metrics, "by_stay_day", 3) == [0, 1, 2]

    assert len(texts) == 1
    rows = list(csv.DictReader(StringIO(next(iter(texts.values())))))
    assert [int(row["day"]) for row in rows] == [1, 2, 3]
    assert [int(row["requests_arriving"]) for row in rows] == [3, 0, 0]
    assert [int(row["requests_covering_night"]) for row in rows] == [0, 1, 2]


def test_complete_episodes_are_averaged_and_partial_is_excluded(logged):
    metrics, texts = logged
    tracker = module.RequestTracker("train")

    record(tracker, 0, 2, [(1, 1)])
    record(tracker, 1, 2, [])
    tracker.end_episode()

    record(tracker, 0, 2, [(1, 1)] * 3)
    record(tracker, 1, 2, [])
    tracker.end_episode()

    # Il terzo episodio è incompleto e non entra nei riepiloghi.
    record(tracker, 0, 2, [(1, 1)] * 100)
    tracker.finalize_training()

    assert curve(metrics, "by_request_day", 2) == [2, 0]
    assert curve(metrics, "by_stay_day", 2) == [0, 2]
    assert len(texts) == 2


def test_all_zero_episode_is_valid(logged):
    metrics, _ = logged
    tracker = module.RequestTracker("train")

    record(tracker, 0, 1, [])
    tracker.end_episode()
    tracker.finalize_training()

    assert curve(metrics, "by_request_day", 1) == [0]
    assert curve(metrics, "by_stay_day", 1) == [0]


def test_missing_day_is_rejected():
    tracker = module.RequestTracker("train", log_to_mlflow=False)
    record(tracker, 0, 3, [])

    with pytest.raises(ValueError, match="out-of-sequence"):
        record(tracker, 2, 3, [])


def test_partial_episode_cannot_be_marked_complete():
    tracker = module.RequestTracker("train", log_to_mlflow=False)
    record(tracker, 0, 2, [])

    with pytest.raises(ValueError, match="incomplete episode"):
        tracker.end_episode()


def test_timestep_metric_counts_requests_not_nights(logged):
    metrics, _ = logged
    tracker = module.RequestTracker("train")

    # Due richieste: una di tre notti, una interamente fuori orizzonte.
    record(tracker, 0, 2, [(1, 3), (5, 2)])
    record(tracker, 1, 2, [(2, 1)])
    tracker.end_episode()
    tracker.log_step_metrics(step=1460)

    key = "num_requests_per_episode/by_training_timesteps/train"
    assert metrics[(key, 1460)] == 3.0

    # Non deve registrare nuovamente gli stessi dati.
    tracker.log_step_metrics(step=1464)
    assert (key, 1464) not in metrics


def test_eval_timestep_metric_averages_only_current_round(logged):
    metrics, _ = logged
    tracker = module.RequestTracker("val")
    key = "num_requests_per_episode/by_training_timesteps/val"

    # Checkpoint 800: due episodi con 2 e 6 richieste.
    for count in [2, 6]:
        record(tracker, 0, 1, [(1, 1)] * count)
        tracker.end_episode()

    assert (key, 800) not in metrics
    tracker.log_step_metrics(step=800)
    assert metrics[(key, 800)] == 4.0

    # Checkpoint 1600: due episodi con 8 e 12 richieste.
    for count in [8, 12]:
        record(tracker, 0, 1, [(1, 1)] * count)
        tracker.end_episode()

    tracker.log_step_metrics(step=1600)
    assert metrics[(key, 1600)] == 10.0


def test_zero_requests_episode_logs_zero(logged):
    metrics, _ = logged
    tracker = module.RequestTracker("train")

    record(tracker, 0, 1, [])
    tracker.end_episode()
    tracker.log_step_metrics(step=4)

    key = "num_requests_per_episode/by_training_timesteps/train"
    assert metrics[(key, 4)] == 0.0


def test_partial_episode_does_not_produce_timestep_metric(logged):
    metrics, _ = logged
    tracker = module.RequestTracker("train")
    key = "num_requests_per_episode/by_training_timesteps/train"

    record(tracker, 0, 2, [(1, 1)])
    tracker.log_step_metrics(step=4)
    assert (key, 4) not in metrics

    tracker.finalize_training()
    tracker.log_step_metrics(step=8)
    assert (key, 8) not in metrics


def test_disabled_logging_clears_pending_episode_totals(logged):
    metrics, _ = logged
    tracker = module.RequestTracker("val", log_to_mlflow=False)
    key = "num_requests_per_episode/by_training_timesteps/val"

    record(tracker, 0, 1, [(1, 1)] * 7)
    tracker.end_episode()
    tracker.log_step_metrics(step=800)
    assert not metrics

    # Non devono riapparire i dati del periodo con logging disabilitato.
    tracker.log_to_mlflow = True
    tracker.log_step_metrics(step=1600)
    assert (key, 1600) not in metrics

    record(tracker, 0, 1, [(1, 1)] * 3)
    tracker.end_episode()
    tracker.log_step_metrics(step=2400)
    assert metrics[(key, 2400)] == 3.0
