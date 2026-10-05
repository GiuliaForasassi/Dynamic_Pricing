import csv
import math
from datetime import date, timedelta
from io import StringIO

import numpy as np
import pytest

from pricing_advertising.plots import price_tracker as module
from pricing_advertising.requests.request import Request

START = date(2025, 12, 29)


@pytest.fixture
def logged(monkeypatch):
    metrics = []
    texts = {}

    def log_metric(key, value, step):
        metrics.append((key, value, step))

    monkeypatch.setattr(module.mlflow, "log_metric", log_metric)
    monkeypatch.setattr(
        module.mlflow,
        "log_text",
        lambda text, path: texts.__setitem__(path, text),
    )
    return metrics, texts


def request(check_in, nights):
    return Request(
        START,
        START + timedelta(days=check_in),
        nights,
        1,
    )


def record(tracker, day, days, requests=(), prices=()):
    tracker.record_day(
        day_index=day,
        start_simulation=START,
        simulation_days=days,
        confirmed_requests=list(requests),
        confirmed_nightly_prices=np.asarray(prices, dtype=float),
    )


def complete(tracker, days, requests=(), prices=()):
    record(tracker, 0, days, requests, prices)
    for day in range(1, days):
        record(tracker, day, days)
    tracker.end_episode()


def values(metrics, key):
    return [(step, value) for name, value, step in metrics if name == key]


def test_nightly_mapping_weighting_and_csv(logged):
    metrics, texts = logged
    tracker = module.PriceTracker("train")

    complete(
        tracker,
        3,
        [request(1, 2), request(2, 1)],
        [100, 300, 200],
    )
    tracker.log_step_metrics(step=1460)
    tracker.finalize_training()

    assert values(metrics, "booked_mean_price/by_training_timesteps/train") == [
        (1460, 200.0)
    ]  # 600 / 3, non (100 + 250) / 2

    daily = values(
        metrics,
        "booked_mean_price/by_stay_day/pooled_completed_episodes/train",
    )
    assert daily[0][0] == 1
    assert math.isnan(daily[0][1])
    assert daily[1:] == [(2, 100.0), (3, 250.0)]

    rows = list(csv.DictReader(StringIO(next(iter(texts.values())))))
    assert [int(row["day"]) for row in rows] == [1, 2, 3]
    assert [int(row["room_nights"]) for row in rows] == [0, 1, 2]
    assert [float(row["booked_revenue"]) for row in rows] == [0, 100, 500]


def test_extended_stays_are_clipped_without_redistribution(logged):
    metrics, _ = logged
    tracker = module.PriceTracker("train")

    # Primo soggiorno: una notte dentro, due fuori.
    # Secondo soggiorno: interamente fuori.
    complete(
        tracker,
        3,
        [request(2, 3), request(5, 1)],
        [100, 200, 300, 900],
    )
    tracker.log_step_metrics(step=1460)

    assert values(metrics, "booked_mean_price/by_training_timesteps/train") == [(1460, 100.0)]


def test_eval_uses_room_night_weights_and_resets_each_round(logged):
    metrics, _ = logged
    tracker = module.PriceTracker("val")
    key = "booked_mean_price/by_training_timesteps/val"

    complete(tracker, 2, [request(1, 1)], [100])
    complete(tracker, 2, [request(1, 1)] * 9, [200] * 9)
    tracker.log_step_metrics(step=800)
    tracker.log_step_metrics(step=800)

    complete(tracker, 2, [request(1, 1)], [50])
    tracker.log_step_metrics(step=1600)

    assert values(metrics, key) == [(800, 190.0), (1600, 50.0)]

    tracker.finalize_training()
    daily = values(
        metrics,
        "booked_mean_price/by_stay_day/pooled_completed_episodes/val",
    )
    assert daily[1][1] == pytest.approx(1950 / 11)


def test_empty_episode_is_nan_but_zero_price_is_valid(logged):
    metrics, texts = logged
    tracker = module.PriceTracker("train")
    key = "booked_mean_price/by_training_timesteps/train"

    complete(tracker, 2)
    tracker.log_step_metrics(step=8)

    complete(tracker, 2, [request(1, 1)], [0])
    tracker.log_step_metrics(step=16)

    result = values(metrics, key)
    assert result[0][0] == 8
    assert math.isnan(result[0][1])
    assert result[1] == (16, 0.0)

    assert len(texts) == 2
    assert any("episode_00000_price.csv" in path for path in texts)
    assert any("episode_00001_price.csv" in path for path in texts)


def test_partial_episode_is_excluded(logged):
    metrics, texts = logged
    tracker = module.PriceTracker("train")

    complete(tracker, 2, [request(1, 1)], [100])
    tracker.log_step_metrics(step=8)

    record(tracker, 0, 2, [request(1, 1)], [9999])
    tracker.finalize_training()
    tracker.log_step_metrics(step=16)

    assert values(metrics, "booked_mean_price/by_training_timesteps/train") == [(8, 100.0)]
    daily = values(
        metrics,
        "booked_mean_price/by_stay_day/pooled_completed_episodes/train",
    )
    assert daily[1] == (2, 100.0)
    assert len(texts) == 1


@pytest.mark.parametrize("bad_day", [0, 2])
def test_duplicate_or_missing_day_is_rejected(bad_day):
    tracker = module.PriceTracker("train", log_to_mlflow=False)
    record(tracker, 0, 3)

    with pytest.raises(ValueError, match="out-of-sequence"):
        record(tracker, bad_day, 3)


def test_incomplete_episode_cannot_be_closed():
    tracker = module.PriceTracker("train", log_to_mlflow=False)
    record(tracker, 0, 2)

    with pytest.raises(ValueError, match="incomplete episode"):
        tracker.end_episode()


@pytest.mark.parametrize("price", [-1, float("nan"), float("inf"), -float("inf")])
def test_invalid_price_is_rejected(price):
    tracker = module.PriceTracker("train", log_to_mlflow=False)

    with pytest.raises(ValueError, match="Invalid prices"):
        record(tracker, 0, 2, [request(1, 1)], [price])


@pytest.mark.parametrize("prices", [[100], [100, 200, 300]])
def test_price_length_mismatch_is_rejected_without_recording_day(prices):
    tracker = module.PriceTracker("train", log_to_mlflow=False)

    with pytest.raises(ValueError, match="Nightly prices not aligned"):
        record(tracker, 0, 3, [request(1, 2)], prices)

    # Il tentativo fallito non deve consumare il giorno.
    record(tracker, 0, 3, [request(1, 2)], [100, 200])


def test_changed_calendar_is_rejected():
    tracker = module.PriceTracker("train", log_to_mlflow=False)
    complete(tracker, 2)

    with pytest.raises(ValueError, match="Calendar changed"):
        record(tracker, 0, 3)


def test_disabled_logging_clears_pending_data(logged):
    metrics, _ = logged
    tracker = module.PriceTracker("val", log_to_mlflow=False)

    complete(tracker, 2, [request(1, 1)], [100])
    tracker.log_step_metrics(step=800)
    assert not metrics

    tracker.log_to_mlflow = True
    tracker.log_step_metrics(step=1600)
    assert not metrics

    complete(tracker, 2, [request(1, 1)], [200])
    tracker.log_step_metrics(step=2400)
    assert values(metrics, "booked_mean_price/by_training_timesteps/val") == [(2400, 200.0)]
