import math

import pytest

from pricing_advertising.plots import acceptance_rate_tracker as module

AcceptanceRateTracker = module.AcceptanceRateTracker


def record(tracker, day, days, counts):
    requests, offers, accepted, confirmed = counts
    tracker.record_day(
        day_index=day,
        simulation_days=days,
        num_requests=requests,
        num_offers=offers,
        num_price_accepted=accepted,
        num_confirmed=confirmed,
    )


@pytest.fixture
def logged(monkeypatch):
    points = []

    def capture(key, value, step):
        points.append((key, value, step))

    monkeypatch.setattr(module.mlflow, "log_metric", capture)
    return points


def point(points, key, step):
    values = [value for name, value, logged_step in points if name == key and logged_step == step]
    assert len(values) == 1
    return values[0]


def test_ratios_use_counts_and_distinguish_acceptance():
    tracker = AcceptanceRateTracker("train", log_to_mlflow=False)

    record(tracker, 0, 2, (1, 1, 1, 1))
    record(tracker, 1, 2, (99, 9, 4, 0))

    counts = tracker.end_episode()

    # Conferme: 1/100, NON media di 100% e 0%.
    assert counts.rates()["confirmation_rate"] == pytest.approx(0.01)

    # Prezzi accettati: (1+4)/(1+9).
    assert counts.rates()["price_acceptance_rate"] == pytest.approx(0.5)


def test_zero_offer_day_keeps_calendar_position(logged):
    tracker = AcceptanceRateTracker("train")

    record(tracker, 0, 3, (1, 1, 1, 1))
    record(tracker, 1, 3, (2, 0, 0, 0))
    record(tracker, 2, 3, (4, 4, 2, 1))
    tracker.end_episode()
    tracker.finalize_training()

    axis = "by_request_day_pooled_completed_episodes/train"

    # Giorno 2: richieste presenti, nessuna conferma e nessuna offerta.
    assert point(logged, f"confirmation_rate/{axis}", 2) == 0.0
    assert math.isnan(point(logged, f"price_acceptance_rate/{axis}", 2))

    # Il terzo giorno NON è spostato al secondo.
    assert point(logged, f"confirmation_rate/{axis}", 3) == pytest.approx(0.25)
    assert point(logged, f"price_acceptance_rate/{axis}", 3) == pytest.approx(0.5)


def test_empty_episode_has_undefined_rates(logged):
    tracker = AcceptanceRateTracker("val")
    record(tracker, 0, 1, (0, 0, 0, 0))
    tracker.end_episode()
    tracker.log_step_metrics(step=800)

    assert len(logged) == 2
    assert all(math.isnan(value) for _, value, _ in logged)


def test_eval_round_pools_episodes_and_resets_checkpoint(logged):
    tracker = AcceptanceRateTracker("val")

    # Primo round: due episodi, 1/1 e 0/99.
    record(tracker, 0, 1, (1, 1, 1, 1))
    tracker.end_episode()
    record(tracker, 0, 1, (99, 99, 0, 0))
    tracker.end_episode()
    tracker.log_step_metrics(step=800)

    key = "confirmation_rate/by_training_timesteps/val"
    assert point(logged, key, 800) == pytest.approx(0.01)

    # Secondo round indipendente: 2/2.
    record(tracker, 0, 1, (2, 2, 2, 2))
    tracker.end_episode()
    tracker.log_step_metrics(step=1600)

    assert point(logged, key, 1600) == 1.0

    # Nessun nuovo episodio: nessun log duplicato.
    previous_length = len(logged)
    tracker.log_step_metrics(step=1600)
    assert len(logged) == previous_length


def test_partial_episode_does_not_enter_daily_summary(logged):
    tracker = AcceptanceRateTracker("train")

    # Episodio completo.
    record(tracker, 0, 2, (1, 1, 1, 1))
    record(tracker, 1, 2, (1, 1, 1, 1))
    tracker.end_episode()
    tracker.log_step_metrics(step=2)

    # Episodio interrotto: non deve abbassare il riepilogo.
    record(tracker, 0, 2, (100, 100, 0, 0))
    tracker.finalize_training()

    key = "confirmation_rate/by_request_day_pooled_completed_episodes/train"
    assert point(logged, key, 1) == 1.0
    assert point(logged, key, 2) == 1.0


def test_incomplete_episode_cannot_be_marked_complete():
    tracker = AcceptanceRateTracker("train", log_to_mlflow=False)
    record(tracker, 0, 2, (1, 1, 1, 1))

    with pytest.raises(ValueError, match="incomplete episode"):
        tracker.end_episode()


def test_missing_day_is_rejected():
    tracker = AcceptanceRateTracker("train", log_to_mlflow=False)
    record(tracker, 0, 3, (1, 1, 1, 1))

    with pytest.raises(ValueError, match="out-of-sequence"):
        record(tracker, 2, 3, (1, 1, 1, 1))


def test_invalid_counts_are_rejected():
    tracker = AcceptanceRateTracker("train", log_to_mlflow=False)

    with pytest.raises(ValueError, match="Acceptance counts are inconsistent"):
        record(tracker, 0, 1, (2, 1, 0, 1))
