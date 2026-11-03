import pickle
from datetime import date, timedelta
from functools import partial
from math import exp, sqrt
from pathlib import Path
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import SubprocVecEnv

from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.plots.episode_metrics import EpisodeMetrics
from pricing_advertising.pricing_model import bar_table
from pricing_advertising.requests.cancellation import CancellationModel
from pricing_advertising.requests.request import Request


@pytest.fixture
def cancellation_model():
    return CancellationModel(reference_prices=(100.0, 200.0, 300.0))


@pytest.mark.parametrize(
    "check_in,prices,odds_factor",
    [
        (date(2025, 3, 1), (100.0,), 1.0),
        (date(2025, 3, 1), (120.0,), exp(-0.2)),
        (date(2025, 7, 1), (100.0,), 0.8),
        (date(2025, 6, 30), (100.0, 100.0), sqrt(0.8)),
    ],
)
def test_probability_matches_agreed_odds(cancellation_model, check_in, prices, odds_factor):
    odds = 0.005 / 0.995 * odds_factor
    assert cancellation_model.probability(
        RoomCategory.CATEGORY1, check_in, prices
    ) == pytest.approx(odds / (1 + odds))


def test_probability_uses_average_price_and_category(cancellation_model):
    for category in RoomCategory:
        price = 100.0 * category.value
        for prices in [(price,), (price * 0.5, price * 1.5)]:
            assert cancellation_model.probability(
                category, date(2025, 3, 1), prices
            ) == pytest.approx(0.005)


def test_probability_decreases_with_price(cancellation_model):
    probabilities = [
        cancellation_model.probability(RoomCategory.CATEGORY1, date(2025, 3, 1), (price,))
        for price in (0.0, 50.0, 100.0, 120.0, 1000.0, 1e308)
    ]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0 <= probability <= 1 for probability in probabilities)


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_probability_endpoints(probability):
    model = CancellationModel(daily_probability=probability)
    assert model.probability(RoomCategory.CATEGORY1, date(2025, 7, 1), (500.0,)) == probability


@pytest.mark.parametrize(
    "kwargs",
    [
        {"daily_probability": -0.1},
        {"daily_probability": 1.1},
        {"daily_probability": float("nan")},
        {"price_sensitivity": -1},
        {"price_sensitivity": float("inf")},
        {"high_season_multiplier": 0},
        {"high_season_multiplier": 1.1},
        {"high_season_multiplier": float("nan")},
        {"reference_prices": (100.0,)},
        {"reference_prices": (100.0, 0.0, 300.0)},
        {"reference_prices": (100.0, float("inf"), 300.0)},
    ],
)
def test_invalid_model_configuration(kwargs):
    with pytest.raises(ValueError):
        CancellationModel(**kwargs)


@pytest.mark.parametrize("prices", [(), (-1.0,), (float("nan"),), (float("inf"),)])
def test_invalid_nightly_prices(cancellation_model, prices):
    with pytest.raises(ValueError, match="Nightly prices"):
        cancellation_model.probability(RoomCategory.CATEGORY1, date(2025, 3, 1), prices)


def test_configuration_is_pickleable_and_copies_references():
    references = [100.0, 200.0, 300.0]
    model = CancellationModel(reference_prices=references)
    references[0] = 1.0
    assert model.reference_prices == (100.0, 200.0, 300.0)
    assert pickle.loads(pickle.dumps(model)) == model


START = date(2025, 3, 1)


@pytest.fixture
def make_cancellation_env(monkeypatch):
    csv_path = Path(__file__).resolve().parents[1] / "data" / "bar_table.csv"
    monkeypatch.setattr(bar_table, "BAR_TABLE", bar_table.load_bar_table_wide_csv(csv_path))

    def make(calendar=None, **kwargs):
        config = {
            "start_simulation": START,
            "end_simulation": START + timedelta(days=5),
            "hotel": Hotel(1, 1, 1),
            "avg_num_requests": 0,
            "seed": 42,
            "bar_update_mode": "never",
            "allow_category_alternative": False,
            "allow_period_alternative": False,
            "collect_eval_artifacts": True,
            "cancellation_model": CancellationModel(daily_probability=1.0),
        }
        config.update(kwargs)
        env = PricingEnv(**config)
        env.acceptance_model.accept = Mock(return_value=True)
        calendar = calendar or {}
        monkeypatch.setattr(
            env.request_generator,
            "fetch_requests_for_date",
            lambda day: np.asarray(calendar.get(day, []), dtype=object),
        )
        env.reset()
        return env

    return make


def step_day(env):
    return env.step(np.zeros(env.action_space.shape, dtype=np.float32))


def test_cancellation_releases_whole_stay_and_refunds_once(make_cancellation_env):
    booking_request = Request(START, START + timedelta(days=3), 2, 1)
    env = make_cancellation_env({START: [booking_request]})
    _, revenue, _, _, booked_info = step_day(env)
    assert revenue > 0
    assert booked_info["step_metrics"]["num_cancelled"] == 0
    np.testing.assert_array_equal(env.occupancy[0, 3:5], [1, 1])
    booked_row = booked_info["eval_artifacts"]["requests"][0]
    _, reward, _, _, cancelled_info = step_day(env)
    event = cancelled_info["eval_artifacts"]["cancellations"][0]
    assert reward == -revenue
    assert event["booking_id"] == booked_row["booking_id"]
    assert event["refund_amount"] == event["reward_debit"] == revenue
    assert event["from_warmup"] is False
    assert booked_row["confirmed"] is True
    assert env.occupancy.sum() == 0
    assert not env._bookings
    assert step_day(env)[1] == 0


def test_cancelled_room_can_be_resold_same_day(make_cancellation_env):
    check_in = START + timedelta(days=3)
    env = make_cancellation_env(
        {
            START: [Request(START, check_in, 2, 1)],
            START + timedelta(days=1): [Request(START + timedelta(days=1), check_in, 2, 1)],
        }
    )
    first_revenue = step_day(env)[1]
    _, reward, _, _, info = step_day(env)
    assert info["step_metrics"]["num_cancelled"] == info["step_metrics"]["num_confirmed"] == 1
    assert info["step_metrics"]["gross_booking_revenue"] == first_revenue
    assert reward == 0
    np.testing.assert_array_equal(env.occupancy[0, 3:5], [1, 1])
    assert info["step_metrics"]["confirmed_booking_ids"] == [1]


def test_tomorrow_checkin_cannot_cancel(make_cancellation_env):
    env = make_cancellation_env({START: [Request(START, START + timedelta(days=1), 2, 1)]})
    step_day(env)
    _, reward, _, _, info = step_day(env)
    assert reward == info["step_metrics"]["num_cancelled"] == 0
    np.testing.assert_array_equal(env.occupancy[0, 1:3], [1, 1])
    assert not env._bookings


def test_cancellation_only_on_last_eligible_day(make_cancellation_env):
    env = make_cancellation_env({START: [Request(START, START + timedelta(days=3), 1, 1)]})
    env.cancellation_rng = Mock()
    env.cancellation_rng.geometric.return_value = 2
    revenue = step_day(env)[1]
    assert step_day(env)[1] == 0
    assert step_day(env)[1] == -revenue
    assert step_day(env)[1] == 0
    env.cancellation_rng.geometric.assert_called_once_with(1.0)
    env.cancellation_rng.random.assert_not_called()


@pytest.mark.parametrize("waiting_days", [2, 4, 1000])
def test_scheduled_cancellations_keep_inventory_until_due(make_cancellation_env, waiting_days):
    check_in = START + timedelta(days=4)
    env = make_cancellation_env(
        {START: [Request(START, check_in, 2, 1)]},
        artifact_snapshot_dates=[START, START + timedelta(days=2)],
    )
    env.cancellation_rng = Mock()
    env.cancellation_rng.geometric.return_value = waiting_days
    observation, revenue, _, _, info = step_day(env)
    snapshot = info["eval_artifacts"]
    assert set(observation) == set(env.observation_space.spaces)
    assert env.occupancy[0, 4:6].tolist() == [1, 1]
    assert snapshot["voided_reservations_by_category"][0][4:6] == (
        [1, 1] if waiting_days < 4 else [0, 0]
    )
    assert info["step_metrics"]["num_cancelled"] == 0
    assert step_day(env)[1] == 0
    _, reward, _, _, info = step_day(env)
    assert reward == (-revenue if waiting_days < 4 else 0)
    assert not np.asarray(info["eval_artifacts"]["voided_reservations_by_category"]).any()
    assert snapshot["occupancy_by_category"][0][4:6] == [1, 1]


def test_snapshot_knows_cancellation_beyond_episode_without_settlement(make_cancellation_env):
    env = make_cancellation_env(
        {START: [Request(START, START + timedelta(days=4), 2, 1)]},
        end_simulation=START + timedelta(days=1),
        artifact_snapshot_dates=[START],
    )
    env.cancellation_rng = Mock()
    env.cancellation_rng.geometric.return_value = 3
    revenue = step_day(env)[1]
    booking = next(iter(env._bookings.values()))
    assert booking.cancellation_day == START + timedelta(days=3)
    assert revenue > 0
    assert step_day(env)[1] == 0
    assert env.occupancy[0, 4:6].tolist() == [1, 1]


def test_snapshot_separates_active_voided_and_already_cancelled(make_cancellation_env):
    from pricing_advertising.plots.eval_artifact_plots import (
        build_snapshot_metrics,
        snapshot_metrics_by_date,
    )

    check_in = START + timedelta(days=4)
    env = make_cancellation_env(
        {START: [Request(START, check_in, 2, guests) for guests in (1, 1, 2, 3)]},
        hotel=Hotel(2, 1, 1),
        artifact_snapshot_dates=[START + timedelta(days=1)],
    )
    env.cancellation_rng = Mock()
    env.cancellation_rng.geometric.side_effect = [1, 3, 3, 1000]
    first = step_day(env)[4]["eval_artifacts"]
    second = step_day(env)[4]["eval_artifacts"]
    metrics = build_snapshot_metrics([first, second], second)
    assert second["num_cancelled"] == 1
    assert not metrics["confirmed"].any()
    np.testing.assert_array_equal(metrics["total_reservations"][:, 3:5], [[1, 1]] * 3)
    np.testing.assert_array_equal(metrics["voided_reservations"][:, 3:5], [[1, 1], [1, 1], [0, 0]])
    serialized = snapshot_metrics_by_date({"steps": [first, second]})
    assert (
        serialized[second["snapshot_start_day"]]["voided_reservations"]
        == metrics["voided_reservations"].tolist()
    )


def test_warmup_pending_cancellations_appear_in_snapshots(make_cancellation_env):
    previous = START - timedelta(days=1)
    env = make_cancellation_env(
        {previous: [Request(previous, START + timedelta(days=3), 2, 1)]},
        warmup_days=1,
        artifact_snapshot_dates=[START],
    )
    env.cancellation_rng = Mock()
    env.cancellation_rng.geometric.return_value = 3
    env.reset()
    _, reward, _, _, info = step_day(env)
    assert reward == 0
    assert info["eval_artifacts"]["voided_reservations_by_category"][0][3:5] == [1, 1]
    step_day(env)
    _, reward, _, _, info = step_day(env)
    assert reward == 0
    assert info["step_metrics"]["num_cancelled"] == 1


def test_invalid_action_and_terminal_step_do_not_cancel(make_cancellation_env):
    env = make_cancellation_env(
        {START: [Request(START, START + timedelta(days=3), 1, 1)]},
        end_simulation=START,
    )
    with pytest.raises(ValueError):
        env.step(np.asarray([float("nan")]))
    step_day(env)
    before = env.occupancy.copy()
    with pytest.raises(gym.error.ResetNeeded):
        step_day(env)
    np.testing.assert_array_equal(env.occupancy, before)
    assert len(env._bookings) == 1


def test_warmup_releases_inventory_without_debiting_reward(make_cancellation_env):
    previous = START - timedelta(days=1)
    check_in = START + timedelta(days=3)
    env = make_cancellation_env(
        {
            previous: [Request(previous, check_in, 2, 1), Request(previous, START, 2, 2)],
            START: [Request(START, check_in, 2, 1)],
        },
        warmup_days=1,
    )
    assert len(env._bookings) == 1
    assert not next(iter(env._bookings.values())).reward_accounted
    np.testing.assert_array_equal(env.occupancy[0, 3:5], [1, 1])
    np.testing.assert_array_equal(env.occupancy[1, :2], [1, 1])
    _, revenue, _, _, info = step_day(env)
    metrics = info["step_metrics"]
    assert metrics["num_cancelled"] == metrics["num_confirmed"] == 1
    assert metrics["refund_total"] > 0
    assert metrics["reward_refund_total"] == 0
    assert revenue == metrics["gross_booking_revenue"] > 0
    assert metrics["cancellations"][0]["from_warmup"] is True
    assert metrics["confirmed_booking_ids"] == [2]
    assert step_day(env)[1] == -revenue
    np.testing.assert_array_equal(env.occupancy[1, :2], [1, 1])
    env.reset(seed=42)
    assert env._next_booking_id == 2
    assert len(env._bookings) == 1
    assert env._warmup_env.cancellation_rng is env.cancellation_rng


@pytest.mark.parametrize("warmup_days", [1, 2])
def test_stay_day_price_excludes_cancelled_bookings_and_keeps_resales(
    make_cancellation_env, warmup_days
):
    previous = START - timedelta(days=warmup_days)
    last_warmup_day = START - timedelta(days=1)
    check_in = START + timedelta(days=3)
    calendar = {
        previous: [Request(previous, check_in, 2, 1)],
        START: [Request(START, check_in, 2, 1)],
        START + timedelta(days=2): [Request(START + timedelta(days=2), check_in, 2, 1)],
    }
    calendar.setdefault(last_warmup_day, []).append(Request(last_warmup_day, START, 2, 2))
    env = make_cancellation_env(calendar, warmup_days=warmup_days)
    tracker = EpisodeMetrics(split="val", log_to_mlflow=False)
    cancellation_origins = []
    resale_prices = None
    for day in range(env.total_days):
        _, _, _, _, info = step_day(env)
        metrics = info["step_metrics"]
        cancellation_origins.extend(event["from_warmup"] for event in metrics["cancellations"])
        if day == 2:
            resale_prices = metrics["confirmed_nightly_prices"].copy()
        tracker.record_day(metrics)
    tracker.end_episode()

    assert cancellation_origins == ([True, False] if warmup_days == 1 else [False])
    assert resale_prices is not None and len(resale_prices) == 2
    warmup_price = bar_table.get_price("CATEGORY2", 1)
    np.testing.assert_array_equal(tracker.price._nights_sum, [1, 1, 0, 1, 1, 0])
    np.testing.assert_allclose(
        tracker.price._revenue_sum, [warmup_price, warmup_price, 0, *resale_prices, 0]
    )


@pytest.mark.parametrize("alternative", ["category", "period"])
def test_recovery_booking_refunds_final_category_and_price(make_cancellation_env, alternative):
    env = make_cancellation_env(
        {START: [Request(START, START + timedelta(days=3), 1, 1) for _ in range(2)]},
        allow_category_alternative=alternative == "category",
        allow_period_alternative=alternative == "period",
    )
    _, revenue, _, _, info = step_day(env)
    bookings = list(env._bookings.values())
    assert len(bookings) == 2
    if alternative == "category":
        assert [booking.category for booking in bookings] == [
            RoomCategory.CATEGORY1,
            RoomCategory.CATEGORY2,
        ]
        assert bookings[0].total_price != bookings[1].total_price
    else:
        assert bookings[0].check_in != bookings[1].check_in
    assert info["eval_artifacts"]["requests"][1]["requested"]["category"] == 1
    _, reward, _, _, info = step_day(env)
    assert reward == -revenue
    assert env.occupancy.sum() == 0
    for event, booking in zip(info["step_metrics"]["cancellations"], bookings, strict=True):
        assert event["category"] == booking.category.value
        assert event["check_in"] == booking.check_in.isoformat()
        assert event["refund_amount"] == booking.total_price


def test_seeded_resets_replay_cancellations_and_clear_state(make_cancellation_env):
    calendar = {
        START + timedelta(days=offset): [
            Request(START + timedelta(days=offset), START + timedelta(days=5), 2, 1)
            for _ in range(10)
        ]
        for offset in range(4)
    }
    env = make_cancellation_env(
        calendar, hotel=Hotel(50, 1, 1), cancellation_model=CancellationModel(daily_probability=0.3)
    )
    episodes = []
    for _ in range(2):
        env.reset(seed=42)
        assert env._next_booking_id == 0
        assert not env._bookings
        steps = [step_day(env)[4]["eval_artifacts"] for _ in range(env.total_days)]
        episodes.append(steps)
    assert episodes[0] == episodes[1]
    assert sum(step["num_cancelled"] for step in episodes[0]) > 0
    assert sum(step["num_cancelled"] for step in episodes[0]) < 40
    state = env.cancellation_rng.bit_generator.state
    env.reset()
    assert env.cancellation_rng.bit_generator.state == state


def test_disabled_model_preserves_legacy_behavior_and_rngs(make_cancellation_env):
    calendar = {START: [Request(START, START + timedelta(days=3), 2, 1)]}
    disabled = make_cancellation_env(calendar, cancellation_model=None)
    zero = make_cancellation_env(
        calendar, cancellation_model=CancellationModel(daily_probability=0)
    )
    for env in (disabled, zero):
        for stream, seed in zip(
            (env.np_rng, env.alternative_rng, env.acceptance_rng),
            np.random.SeedSequence(42).spawn(3),
            strict=True,
        ):
            np.testing.assert_array_equal(stream.random(10), np.random.default_rng(seed).random(10))
    initial_state = zero.cancellation_rng.bit_generator.state
    for _ in range(disabled.total_days):
        first = step_day(disabled)
        second = step_day(zero)
        assert first[1:4] == second[1:4]
        assert first[4]["eval_artifacts"] == second[4]["eval_artifacts"]
        np.testing.assert_array_equal(disabled.occupancy, zero.occupancy)
    assert zero.cancellation_rng.bit_generator.state == initial_state


def test_cancelled_booking_uses_frozen_prices_and_post_cancel_snapshot(make_cancellation_env):
    check_in = START + timedelta(days=3)
    env = make_cancellation_env(
        {START: [Request(START, check_in, 2, 1)]},
        artifact_snapshot_dates=[START + timedelta(days=1), check_in],
    )
    revenue = step_day(env)[1]
    env.bar_levels.fill(env._max_bar_level)
    _, reward, _, _, info = step_day(env)
    assert reward == -revenue
    payload = info["eval_artifacts"]
    assert payload["bar_history"][check_in.isoformat()]["occupancy_before"] == [0, 0, 0]
    assert not np.asarray(payload["occupancy_by_category"]).any()


def test_underflow_is_rejected_without_refund(make_cancellation_env):
    env = make_cancellation_env({START: [Request(START, START + timedelta(days=3), 2, 1)]})
    step_day(env)
    env.occupancy[0, 3] = 0
    with pytest.raises(RuntimeError, match="unoccupied"):
        step_day(env)
    assert len(env._bookings) == 1
    assert env.occupancy[0, 4] == 1


def test_subprocess_environment_matches_seeded_local_cancellations(make_cancellation_env):
    make_cancellation_env()
    factory = partial(
        PricingEnv,
        start_simulation=START,
        end_simulation=START + timedelta(days=5),
        hotel=Hotel(5, 5, 5),
        avg_num_requests=10,
        seed=42,
        warmup_days=2,
        bar_update_mode="never",
        cancellation_model=CancellationModel(daily_probability=1),
    )
    local = factory()
    vector = SubprocVecEnv([factory], start_method="fork")
    cancellations = 0
    try:
        local.reset()
        vector.reset()
        action = np.zeros(local.action_space.shape, dtype=np.float32)
        for _ in range(local.total_days):
            _, local_reward, terminated, _, local_info = local.step(action)
            _, rewards, dones, infos = vector.step(action[None])
            assert rewards[0] == pytest.approx(local_reward)
            assert dones[0] == terminated
            assert (
                infos[0]["step_metrics"]["cancellations"]
                == local_info["step_metrics"]["cancellations"]
            )
            cancellations += local_info["step_metrics"]["num_cancelled"]
        assert cancellations > 0
    finally:
        local.close()
        vector.close()
