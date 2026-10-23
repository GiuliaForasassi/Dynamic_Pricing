import math
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.plots import alternatives_plots
from pricing_advertising.plots import alternatives_tracker as module
from pricing_advertising.plots.episode_metrics import EpisodeMetrics
from pricing_advertising.pricing_model import bar_table
from pricing_advertising.requests.request import (
    ALTERNATIVE_TYPES,
    OFFER_TYPES,
    ORIGINAL_OFFER,
    HotelOffer,
    Request,
)

AlternativesTracker = module.AlternativesTracker


# ---------------- Offer classification ----------------
@pytest.mark.parametrize(
    "category,shift,nights,expected",
    [
        (RoomCategory.CATEGORY1, 0, 3, ORIGINAL_OFFER),
        (RoomCategory.CATEGORY2, 0, 3, "cat_1_to_2"),
        (RoomCategory.CATEGORY3, 0, 3, "cat_1_to_3"),
        (RoomCategory.CATEGORY1, 1, 2, "period_start_later_shorter"),
        (RoomCategory.CATEGORY1, 0, 2, "period_end_earlier"),
        (RoomCategory.CATEGORY1, -1, 3, "period_shift_earlier"),
        (RoomCategory.CATEGORY1, 1, 3, "period_shift_later"),
    ],
)
def test_offer_type(category, shift, nights, expected):
    request = Request(date(2025, 1, 1), date(2025, 1, 10), 3, 1)
    offer = HotelOffer(request, category, request.t_reservation + timedelta(days=shift), nights)
    assert offer.offer_type == expected
    assert expected in OFFER_TYPES


def test_category_2_alternative_type_is_named_from_the_request():
    request = Request(date(2025, 1, 1), date(2025, 1, 10), 3, 2)
    offer = HotelOffer(request, RoomCategory.CATEGORY3, request.t_reservation, 3)
    assert offer.offer_type == "cat_2_to_3"


def test_unsupported_offer_is_rejected():
    request = Request(date(2025, 1, 1), date(2025, 1, 10), 3, 1)
    offer = HotelOffer(request, RoomCategory.CATEGORY2, request.t_reservation, 2)
    with pytest.raises(ValueError, match="both category and period"):
        _ = offer.offer_type


# ---------------- Environment statistics ----------------
@pytest.fixture
def make_env(monkeypatch):
    csv_path = Path(__file__).resolve().parents[1] / "data" / "bar_table.csv"
    monkeypatch.setattr(bar_table, "BAR_TABLE", bar_table.load_bar_table_wide_csv(csv_path))
    environments = []

    def factory(**kwargs):
        env = PricingEnv(
            date(2025, 1, 1),
            date(2025, 1, 4),
            Hotel(1, 1, 1),
            avg_num_requests=5,
            seed=42,
            **kwargs,
        )
        environments.append(env)
        return env

    yield factory
    for env in environments:
        env.close()


def step(env, requests, monkeypatch):
    env.request_generator.requests_by_day = {env.today: requests}
    monkeypatch.setattr(env.acceptance_model, "accept", Mock(return_value=True))
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape))
    return reward, info["step_metrics"]


def test_failed_booking_is_recovered_with_a_recorded_alternative(make_env, monkeypatch):
    env = make_env(allow_category_alternative=True, allow_period_alternative=False)
    check_in = env.today + timedelta(days=2)
    requests = [Request(env.today, check_in, 1, 1), Request(env.today, check_in, 1, 1)]

    reward, metrics = step(env, requests, monkeypatch)

    stats = metrics["offer_stats"]
    assert [
        (e["type"], e["stage"], e["price_accepted"], e["confirmed"]) for e in stats["events"]
    ] == [
        (ORIGINAL_OFFER, "initial", True, True),
        (ORIGINAL_OFFER, "initial", True, False),
        ("cat_1_to_2", "recovery", True, True),
    ]
    assert stats["num_outside_booking_window"] == stats["num_unavailable"] == 0
    assert {e["check_in_index"] for e in stats["events"]} == {2}

    # The recorded revenue is the reward of the step.
    assert sum(e["price"] for e in stats["events"]) == pytest.approx(reward)
    recovered = stats["events"][2]
    assert recovered["reference_price"] == pytest.approx(bar_table.get_price("CATEGORY1", 1))
    assert recovered["price"] == pytest.approx(bar_table.get_price("CATEGORY2", 1))


def test_requests_without_offer_are_classified(make_env, monkeypatch):
    env = make_env(allow_category_alternative=False, allow_period_alternative=False)
    check_in = env.today + timedelta(days=2)
    env.occupancy[RoomCategory.CATEGORY3.value - 1, :] = 1
    requests = [
        Request(env.today, env.today, 1, 1),  # outside the booking window
        Request(env.today, check_in, 1, 3),  # full category, alternatives disabled
        Request(env.today, check_in, 1, 1),
    ]

    _, metrics = step(env, requests, monkeypatch)

    stats = metrics["offer_stats"]
    assert stats["num_outside_booking_window"] == 1
    assert stats["num_unavailable"] == 1
    assert [e["type"] for e in stats["events"]] == [ORIGINAL_OFFER]


def test_period_alternative_before_pricing_is_recorded(make_env, monkeypatch):
    env = make_env(allow_category_alternative=False, allow_period_alternative=True)
    check_in = env.today + timedelta(days=1)
    env.occupancy[RoomCategory.CATEGORY1.value - 1, 1] = 1  # day of the requested check-in is full
    requests = [Request(env.today, check_in, 2, 1)]

    _, metrics = step(env, requests, monkeypatch)

    events = metrics["offer_stats"]["events"]
    assert [(e["type"], e["stage"]) for e in events] == [("period_start_later_shorter", "initial")]
    assert events[0]["check_in_index"] == 1


def test_tracker_state_follows_the_environment(make_env, monkeypatch):
    env = make_env(allow_category_alternative=True, allow_period_alternative=False)
    check_in = env.today + timedelta(days=2)
    requests = [Request(env.today, check_in, 1, 1), Request(env.today, check_in, 1, 1)]
    _, metrics = step(env, requests, monkeypatch)

    tracker = AlternativesTracker("train", log_to_mlflow=False)
    tracker.record_day(
        day_index=0,
        simulation_days=1,
        num_requests=metrics["num_requests"],
        occupancy_pct=metrics["occupancy_pct"],
        offer_stats=metrics["offer_stats"],
    )
    tracker.end_episode()


# ---------------- Tracker ----------------
def event(
    offer_type=ORIGINAL_OFFER,
    *,
    stage="initial",
    accepted=True,
    confirmed=True,
    price=100.0,
    reference=None,
    check_in=0,
):
    return {
        "type": offer_type,
        "stage": stage,
        "check_in_index": check_in,
        "price_accepted": accepted,
        "confirmed": confirmed,
        "price": price if confirmed else 0.0,
        "reference_price": (price if reference is None else reference) if confirmed else 0.0,
    }


def record(tracker, day, days, events, *, outside=0, unavailable=0, occupancy=50.0):
    initial = sum(1 for e in events if e["stage"] == "initial")
    tracker.record_day(
        day_index=day,
        simulation_days=days,
        num_requests=initial + outside + unavailable,
        occupancy_pct=occupancy,
        offer_stats={
            "events": events,
            "num_outside_booking_window": outside,
            "num_unavailable": unavailable,
        },
    )


@pytest.fixture
def logged(monkeypatch):
    points = []
    monkeypatch.setattr(
        module.mlflow, "log_metric", lambda key, value, step: points.append((key, value, step))
    )
    return points


@pytest.fixture
def figures(monkeypatch):
    out = {}
    monkeypatch.setattr(module.mlflow, "log_figure", lambda fig, path: out.update({path: fig}))
    return out


def value(points, key, step):
    values = [v for name, v, s in points if name == key and s == step]
    assert len(values) == 1, key
    return values[0]


SUFFIX = "by_training_timesteps/train"


def test_counts_types_and_rates(logged):
    tracker = AlternativesTracker("train")
    record(
        tracker,
        0,
        2,
        [
            event(),
            event(),
            event("cat_1_to_2", price=80.0, reference=100.0),
            event("period_shift_later", accepted=False, confirmed=False),
        ],
        outside=1,
        unavailable=2,
    )
    record(
        tracker,
        1,
        2,
        [
            event(),
            event("cat_1_to_2", stage="recovery", accepted=True, confirmed=False),
            event("period_end_earlier", price=50.0, reference=60.0),
        ],
    )
    tracker.end_episode()
    tracker.log_step_metrics(step=1000)

    assert value(logged, f"alternatives_proposed/alternative/{SUFFIX}", 1000) == 4
    assert value(logged, f"alternatives_proposed/category/{SUFFIX}", 1000) == 2
    assert value(logged, f"alternatives_proposed/period/{SUFFIX}", 1000) == 2
    assert value(logged, f"alternatives_proposed/after_failed_booking/{SUFFIX}", 1000) == 1
    assert value(logged, f"alternatives_proposed_by_type/cat_1_to_2/{SUFFIX}", 1000) == 2
    assert value(logged, f"alternatives_proposed_by_type/cat_1_to_3/{SUFFIX}", 1000) == 0
    # Requests: day 0 has 4 initial offers + 1 outside window + 2 unavailable, day 1 has 2 initial offers.
    assert value(logged, f"alternative_proposal_rate/{SUFFIX}", 1000) == pytest.approx(4 / 9)

    # Alternatives: 4 proposed, 3 price accepted, 2 confirmed.
    assert value(
        logged, f"offer_price_acceptance_rate/alternative/{SUFFIX}", 1000
    ) == pytest.approx(3 / 4)
    assert value(logged, f"offer_confirmation_rate/alternative/{SUFFIX}", 1000) == pytest.approx(
        2 / 4
    )
    assert value(logged, f"offer_confirmation_rate/original/{SUFFIX}", 1000) == 1.0

    # Revenue: originals 300, alternatives 130.
    assert value(logged, f"alternative_revenue/{SUFFIX}", 1000) == 130.0
    assert value(logged, f"alternative_revenue_share/{SUFFIX}", 1000) == pytest.approx(130 / 430)
    assert value(logged, f"alternative_price_ratio/{SUFFIX}", 1000) == pytest.approx(130 / 160)
    assert value(logged, f"requests_without_offer/no_availability/{SUFFIX}", 1000) == 2
    assert value(logged, f"requests_without_offer/outside_window/{SUFFIX}", 1000) == 1


def test_run_without_alternatives_skips_undefined_rates(logged):
    tracker = AlternativesTracker("train")
    record(tracker, 0, 1, [event(), event(accepted=False, confirmed=False)], unavailable=3)
    tracker.end_episode()
    tracker.log_step_metrics(step=10)

    assert value(logged, f"alternatives_proposed/alternative/{SUFFIX}", 10) == 0
    assert value(logged, f"alternative_revenue/{SUFFIX}", 10) == 0
    assert value(logged, f"requests_without_offer/no_availability/{SUFFIX}", 10) == 3
    assert value(logged, f"offer_price_acceptance_rate/original/{SUFFIX}", 10) == 0.5
    logged_keys = {key for key, _, _ in logged}
    assert not any(math.isnan(v) for _, v, _ in logged)
    assert f"alternative_proposal_rate/{SUFFIX}" in logged_keys
    assert f"offer_price_acceptance_rate/alternative/{SUFFIX}" not in logged_keys
    assert f"alternative_price_ratio/{SUFFIX}" not in logged_keys


def test_log_step_metrics_averages_pending_episodes_and_resets(logged):
    tracker = AlternativesTracker("train")
    record(tracker, 0, 1, [event("cat_1_to_2")])
    tracker.end_episode()
    record(tracker, 0, 1, [event("cat_1_to_2"), event("cat_1_to_2")])
    tracker.end_episode()
    tracker.log_step_metrics(step=1)

    assert value(logged, f"alternatives_proposed/alternative/{SUFFIX}", 1) == 1.5
    length = len(logged)
    tracker.log_step_metrics(step=1)
    assert len(logged) == length


def test_partial_episode_does_not_enter_figures(figures):
    tracker = AlternativesTracker("train")
    record(tracker, 0, 2, [event("cat_1_to_2")])
    record(tracker, 1, 2, [])
    tracker.end_episode()
    # Interrupted episode
    record(tracker, 0, 2, [event("cat_1_to_2")] * 10)
    tracker.finalize_training()

    assert set(figures) == {
        f"alternatives/train/{name}.png"
        for name in (
            "by_checkin_day",
            "by_type",
            "funnel_by_type",
            "request_funnel",
            "revenue_impact",
        )
    }
    day_axis = figures["alternatives/train/by_checkin_day.png"].axes[0]
    assert sum(patch.get_height() for patch in day_axis.patches) == 1


def test_by_checkin_day_ignores_checkins_after_the_simulation(figures):
    tracker = AlternativesTracker("train")
    record(
        tracker,
        0,
        2,
        [event("cat_1_to_2", check_in=1), event("cat_1_to_2", check_in=7)],
    )
    record(tracker, 1, 2, [])
    tracker.end_episode()
    tracker.finalize_training()

    day_axis = figures["alternatives/train/by_checkin_day.png"].axes[0]
    assert sum(patch.get_height() for patch in day_axis.patches) == 1
    type_axis = figures["alternatives/train/by_type.png"].axes[0]
    assert sum(patch.get_height() for patch in type_axis.patches[:7]) == 2


def test_figures_survive_a_run_without_alternatives(figures):
    tracker = AlternativesTracker("val")
    record(tracker, 0, 1, [event()], unavailable=1)
    tracker.end_episode()
    tracker.finalize_training()

    assert len(figures) == 5


def test_mlflow_disabled_logs_nothing(logged, figures):
    tracker = AlternativesTracker("train", log_to_mlflow=False)
    record(tracker, 0, 1, [event("cat_1_to_2")])
    tracker.end_episode()
    tracker.log_step_metrics(step=1)
    tracker.finalize_training()

    assert not logged
    assert not figures


def test_validation_errors():
    tracker = AlternativesTracker("train", log_to_mlflow=False)
    record(tracker, 0, 3, [event()])

    with pytest.raises(ValueError, match="out-of-sequence"):
        record(tracker, 2, 3, [event()])
    with pytest.raises(ValueError, match="incomplete episode"):
        tracker.end_episode()

    other = AlternativesTracker("train", log_to_mlflow=False)
    with pytest.raises(ValueError, match="inconsistent with the number of requests"):
        other.record_day(
            day_index=0,
            simulation_days=1,
            num_requests=5,
            occupancy_pct=0.0,
            offer_stats={
                "events": [event()],
                "num_outside_booking_window": 0,
                "num_unavailable": 0,
            },
        )
    with pytest.raises(ValueError, match="accepted price"):
        record(other, 0, 1, [event(accepted=False, confirmed=True)])


def test_episode_metrics_feeds_the_tracker():
    metrics = EpisodeMetrics(split="val", log_to_mlflow=False)
    assert metrics.alternatives in metrics._trackers


def test_layout_constants_match_offer_types():
    assert len(OFFER_TYPES) == 1 + len(ALTERNATIVE_TYPES) == 8
    assert alternatives_plots.ORIGINAL_ROW == 0
    assert alternatives_plots.ALTERNATIVE_ROWS == list(range(1, 8))
