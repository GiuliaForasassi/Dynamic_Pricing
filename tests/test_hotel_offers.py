import pickle
import random
import struct
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from pricing_advertising.acceptance.logistic_function import LogisticAcceptanceModel
from pricing_advertising.acceptance.simple_function import SimpleAcceptanceModel
from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.pricing_model import bar_table
from pricing_advertising.requests.request import HotelOffer, Request
from pricing_advertising.requests.request_generator import RequestGenerator
from pricing_advertising.utils.constants import (
    LOGISTIC_ACCEPTANCE_AVG_PRICES,
    LOGISTIC_ACCEPTANCE_HIGH_SEASON_MULTIPLIERS,
    LOGISTIC_ACCEPTANCE_STDS,
)


@pytest.fixture
def booking_request():
    return Request(date(2025, 1, 1), date(2025, 1, 3), 3, 1)


@pytest.fixture(params=["logistic", "simple_low_price", "simple_high_price"])
def acceptance_case(request):
    if request.param == "logistic":
        model = LogisticAcceptanceModel(1.0, {1: 100.0}, {1: 10.0})
        return model, 300.0, 0.5
    model = SimpleAcceptanceModel({RoomCategory.CATEGORY1: 400.0})
    if request.param == "simple_low_price":
        return model, 300.0, 0.8
    return model, 600.0, 0.2


@pytest.mark.parametrize(
    "category,day_shift,nights,penalized",
    [
        (RoomCategory.CATEGORY1, 0, 3, False),
        (RoomCategory.CATEGORY2, 0, 3, False),
        (RoomCategory.CATEGORY1, 1, 3, True),
        (RoomCategory.CATEGORY1, 0, 2, True),
        (RoomCategory.CATEGORY2, 1, 2, True),
    ],
)
def test_acceptance_penalizes_only_period(
    booking_request, acceptance_case, category, day_shift, nights, penalized
):
    request = booking_request
    original = vars(request).copy()
    model, total_price, base_probability = acceptance_case
    assert isinstance(model.rng, np.random.Generator)
    model.rng = Mock()
    model.rng.normal.return_value = 100.0
    model.rng.random.return_value = base_probability * 0.9
    offer = HotelOffer(request, category, request.t_reservation + timedelta(days=day_shift), nights)
    price = total_price / request.days_stay * nights
    assert model.accept(price, offer) == (not penalized)
    assert model.accept_daily_price(price / nights, offer) == (not penalized)
    model.rng.random.return_value = base_probability * 0.79
    assert model.accept(price, offer)
    model.rng.random.return_value = base_probability * 1.01
    assert not model.accept(price, offer)
    assert vars(request) == original


@pytest.mark.parametrize(
    "available_category", [RoomCategory.CATEGORY2, RoomCategory.CATEGORY3, None]
)
def test_alternative_room_preserves_request(booking_request, available_category):
    request = booking_request
    env = PricingEnv.__new__(PricingEnv)
    env.check_hotel_availability = Mock(
        side_effect=lambda offer: offer.category == available_category
    )
    original = vars(request).copy()
    offer = env._check_alternative_room(request)
    if available_category is None:
        assert offer is None
    else:
        assert isinstance(offer, HotelOffer)
        assert offer.request is request
        assert offer.category == available_category
        assert offer.t_reservation == request.t_reservation
        assert offer.days_stay == request.days_stay
        assert not offer.period_changed
    assert vars(request) == original


def test_acceptance_daily_and_total_prices_match(booking_request, acceptance_case):
    request = booking_request
    model, price, _ = acceptance_case
    offer = HotelOffer(request, request.category, request.t_reservation, request.days_stay)
    model.rng = np.random.default_rng(42)
    total_results = [model.accept(price, offer) for _ in range(30)]
    model.rng = np.random.default_rng(42)
    daily_results = [model.accept_daily_price(price / offer.days_stay, offer) for _ in range(30)]
    assert daily_results == total_results


@pytest.mark.parametrize(
    "check_in,nights,expected_mean",
    [
        (date(2025, 4, 10), 3, 100.0),
        (date(2025, 8, 10), 3, 120.0),
        (date(2025, 6, 30), 2, 110.0),
        (date(2025, 9, 30), 4, 105.0),
        (date(2025, 6, 30), 1, 100.0),
        (date(2025, 12, 31), 2, 120.0),
        (None, 1, 100.0),
    ],
)
def test_reference_price_uses_stay_nights(check_in, nights, expected_mean):
    model = LogisticAcceptanceModel(1.0, {1: 100.0}, {1: 10.0}, high_season_multipliers={1: 1.2})
    model.rng = Mock()
    model.rng.normal.return_value = 123.0

    assert (
        model.generate_reference_price(
            RoomCategory.CATEGORY1, t_reservation=check_in, days_stay=nights
        )
        == 123.0
    )
    model.rng.normal.assert_called_once_with(loc=pytest.approx(expected_mean), scale=10.0)


@pytest.mark.parametrize("request_day", [date(2025, 1, 1), date(2025, 4, 1)])
@pytest.mark.parametrize("alternative", [False, True])
def test_seasonal_acceptance_uses_offered_period(request_day, alternative):
    model = LogisticAcceptanceModel(1.0, {1: 100.0}, {1: 10.0}, high_season_multipliers={1: 1.2})
    booking = Request(request_day, date(2025, 6, 30), 2, 1)
    check_in = date(2025, 8, 1) if alternative else booking.t_reservation
    nights = 3 if alternative else booking.days_stay
    offer = HotelOffer(booking, booking.category, check_in, nights)
    model.rng = Mock()
    model.rng.normal.side_effect = lambda loc, scale: loc
    model.rng.random.return_value = 0.39 if alternative else 0.49
    daily_price = 120.0 if alternative else 110.0

    assert model.accept(daily_price * nights, offer)
    assert model.accept_daily_price(daily_price, offer)
    model.rng.normal.assert_called_with(loc=pytest.approx(daily_price), scale=10.0)
    model.rng.random.return_value = 0.41 if alternative else 0.51
    assert not model.accept(daily_price * nights, offer)


@pytest.mark.parametrize("multiplier", [0.0, -1.0, float("inf"), float("nan")])
def test_reference_price_rejects_invalid_seasonal_multiplier(multiplier):
    with pytest.raises(ValueError, match="finite and positive"):
        LogisticAcceptanceModel(1.0, {1: 100.0}, {1: 10.0}, high_season_multipliers={1: multiplier})


def test_seasonal_reference_price_preserves_category_and_seed():
    model = LogisticAcceptanceModel(
        1.0,
        {1: 100.0, 2: 200.0},
        {1: 10.0, 2: 20.0},
        high_season_multipliers={1: 1.2, 2: 1.5},
    )
    for category, expected_shift in [(RoomCategory.CATEGORY1, 20), (RoomCategory.CATEGORY2, 100)]:
        model.rng = np.random.default_rng(42)
        low_season = model.generate_reference_price(category, t_reservation=date(2025, 4, 1))
        model.rng = np.random.default_rng(42)
        high_season = model.generate_reference_price(category, t_reservation=date(2025, 8, 1))
        assert high_season - low_season == pytest.approx(expected_shift)


@pytest.mark.parametrize("multipliers", [None, {}, {1: 1.0}, {2: 1.5}])
def test_neutral_seasonality_preserves_reference_draw(multipliers):
    model = LogisticAcceptanceModel(1.0, {1: 100.0}, {1: 10.0}, high_season_multipliers=multipliers)
    model.rng = np.random.default_rng(42)
    expected = model.generate_reference_price(RoomCategory.CATEGORY1)
    model.rng = np.random.default_rng(42)
    assert (
        model.generate_reference_price(
            RoomCategory.CATEGORY1, t_reservation=date(2025, 8, 1), days_stay=3
        )
        == expected
    )


def test_high_season_increases_acceptance_at_same_price():
    model = LogisticAcceptanceModel(1.0, {1: 100.0}, {1: 10.0}, high_season_multipliers={1: 1.2})
    model.rng = Mock()
    model.rng.normal.side_effect = lambda loc, scale: loc
    model.rng.random.return_value = 0.5
    for check_in, expected in [(date(2025, 4, 1), False), (date(2025, 8, 1), True)]:
        booking = Request(date(2025, 1, 1), check_in, 2, 1)
        offer = HotelOffer(booking, booking.category, check_in, 2)
        assert model.accept(220.0, offer) == expected


def test_seasonal_reference_price_remains_nonnegative():
    model = LogisticAcceptanceModel(1.0, {1: 100.0}, {1: 10.0}, high_season_multipliers={1: 1.2})
    model.rng = Mock()
    model.rng.normal.return_value = -1.0
    assert (
        model.generate_reference_price(RoomCategory.CATEGORY1, t_reservation=date(2025, 8, 1))
        == 0.0
    )


@pytest.mark.parametrize("nights", [0, -1])
def test_reference_price_requires_positive_stay(nights):
    model = LogisticAcceptanceModel(1.0, {1: 100.0}, {1: 10.0})
    with pytest.raises(ValueError, match="at least one night"):
        model.generate_reference_price(RoomCategory.CATEGORY1, days_stay=nights)


@pytest.mark.parametrize("day_shift,nights", [(1, 2), (0, 2), (-1, 3), (1, 3)])
def test_alternative_period_checks_offered_dates(booking_request, day_shift, nights):
    env = PricingEnv.__new__(PricingEnv)
    env.alternative_rng = np.random.default_rng(42)
    env.extended_end_simulation = date(2025, 1, 20)
    check_in = booking_request.t_reservation + timedelta(days=day_shift)
    env.check_hotel_availability = Mock(
        side_effect=lambda offer: (offer.t_reservation, offer.days_stay) == (check_in, nights)
    )
    original = vars(booking_request).copy()

    offer = env._check_alternative_period(booking_request, booking_request.t_request)

    assert isinstance(offer, HotelOffer)
    assert offer.request is booking_request
    assert offer.t_reservation == check_in
    assert offer.days_stay == nights
    assert offer.period_changed
    assert vars(booking_request) == original


def test_alternative_period_respects_calendar_boundaries():
    env = PricingEnv.__new__(PricingEnv)
    env.alternative_rng = np.random.default_rng(42)
    env.extended_end_simulation = date(2025, 1, 3)
    env.check_hotel_availability = Mock(return_value=True)
    request = Request(date(2025, 1, 1), date(2025, 1, 2), 1, 1)

    assert env._check_alternative_period(request, request.t_request) is None
    env.check_hotel_availability.assert_not_called()


@pytest.fixture
def make_env(monkeypatch):
    csv_path = Path(__file__).resolve().parents[1] / "data" / "bar_table.csv"
    monkeypatch.setattr(bar_table, "BAR_TABLE", bar_table.load_bar_table_wide_csv(csv_path))
    environments = []

    def factory(seed=42, **kwargs):
        env = PricingEnv(
            date(2025, 1, 1),
            date(2025, 1, 4),
            Hotel(1, 1, 1),
            avg_num_requests=5,
            seed=seed,
            **kwargs,
        )
        environments.append(env)
        return env

    yield factory
    for env in environments:
        env.close()


@pytest.mark.parametrize("category", list(RoomCategory))
def test_default_env_uses_seasonal_acceptance(make_env, category):
    env = make_env()
    model = env.acceptance_model
    model.rng = Mock()
    model.rng.normal.return_value = 100.0
    model.rng.random.return_value = 0.0
    booking = Request(date(2025, 1, 1), date(2025, 6, 30), 2, category.value)
    offer = HotelOffer(booking, category, booking.t_reservation, booking.days_stay)

    assert model.accept(200.0, offer)
    multiplier = LOGISTIC_ACCEPTANCE_HIGH_SEASON_MULTIPLIERS[category.value]
    model.rng.normal.assert_called_once_with(
        loc=pytest.approx(LOGISTIC_ACCEPTANCE_AVG_PRICES[category.value] * (1 + multiplier) / 2),
        scale=LOGISTIC_ACCEPTANCE_STDS[category.value],
    )


@pytest.mark.parametrize("allow_category", [False, True])
@pytest.mark.parametrize("allow_period", [False, True])
@pytest.mark.parametrize("category_available", [False, True])
@pytest.mark.parametrize("period_available", [False, True])
def test_alternative_flags_control_initial_offers(
    make_env, monkeypatch, allow_category, allow_period, category_available, period_available
):
    env = make_env(
        allow_category_alternative=allow_category,
        allow_period_alternative=allow_period,
    )
    env.reset(seed=42)
    assert env.allow_category_alternative is allow_category
    assert env.allow_period_alternative is allow_period
    booking = Request(env.today, env.today + timedelta(days=2), 1, 1)
    original = vars(booking).copy()
    env.occupancy[0, 2] = 1
    if not category_available:
        env.occupancy[1:, 2] = 1
    if not period_available:
        env.occupancy[0, [1, 3]] = 1
    category_search = Mock(wraps=env._check_alternative_room)
    period_search = Mock(wraps=env._check_alternative_period)
    monkeypatch.setattr(env, "_check_alternative_room", category_search)
    monkeypatch.setattr(env, "_check_alternative_period", period_search)
    rng_state = deepcopy(env.alternative_rng.bit_generator.state)

    mask, offers = env._process_requests(np.asarray([booking], dtype=object), env.today)

    category_found = allow_category and category_available
    expected_offer = category_found or (allow_period and period_available)
    assert mask.tolist() == [expected_offer]
    assert offers.shape == (int(expected_offer),)
    assert category_search.call_count == int(allow_category)
    assert period_search.call_count == int(allow_period and not category_found)
    if expected_offer:
        offer = offers[0]
        assert offer.request is booking
        assert offer.category == (
            RoomCategory.CATEGORY2 if category_found else RoomCategory.CATEGORY1
        )
        assert offer.period_changed == (not category_found)
        assert env.check_hotel_availability(offer)
    if not allow_period:
        assert env.alternative_rng.bit_generator.state == rng_state
    assert vars(booking) == original


@pytest.mark.parametrize("allow_category", [False, True])
@pytest.mark.parametrize("allow_period", [False, True])
def test_alternative_flags_preserve_available_original_offer(
    make_env, monkeypatch, allow_category, allow_period
):
    env = make_env(
        allow_category_alternative=allow_category,
        allow_period_alternative=allow_period,
    )
    booking = Request(env.today, env.today + timedelta(days=2), 1, 1)
    category_search = Mock()
    period_search = Mock()
    monkeypatch.setattr(env, "_check_alternative_room", category_search)
    monkeypatch.setattr(env, "_check_alternative_period", period_search)

    mask, offers = env._process_requests(np.asarray([booking], dtype=object), env.today)

    assert mask.tolist() == [True]
    assert offers.tolist() == [HotelOffer(booking, booking.category, booking.t_reservation, 1)]
    category_search.assert_not_called()
    period_search.assert_not_called()


@pytest.mark.parametrize("allow_category", [False, True])
@pytest.mark.parametrize("allow_period", [False, True])
def test_alternative_flags_control_confirmation_conflicts(
    make_env, monkeypatch, allow_category, allow_period
):
    env = make_env(
        collect_eval_artifacts=True,
        allow_category_alternative=allow_category,
        allow_period_alternative=allow_period,
    )
    booking = Request(env.today, env.today + timedelta(days=2), 1, 1)
    env.request_generator.requests_by_day = {env.today: [booking, deepcopy(booking)]}
    accept = Mock(return_value=True)
    monkeypatch.setattr(env.acceptance_model, "accept", accept)

    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape))

    recovered = allow_category or allow_period
    assert accept.call_count == 2 + int(recovered)
    assert info["step_metrics"]["num_requests"] == info["step_metrics"]["num_offers"] == 2
    assert info["step_metrics"]["num_confirmed"] == 1 + int(recovered)
    assert env.occupancy.sum() == 1 + int(recovered)
    expected_reward = bar_table.get_price("CATEGORY1", 1)
    if recovered:
        expected_reward += bar_table.get_price("CATEGORY2" if allow_category else "CATEGORY1", 1)
    assert reward == pytest.approx(expected_reward)
    rows = info["eval_artifacts"]["requests"]
    assert [len(row["offer_attempts"]) for row in rows] == [1, 1 + int(recovered)]
    assert rows[1]["confirmed"] == recovered
    if recovered:
        assert rows[1]["category"] == (2 if allow_category else 1)
        assert rows[1]["period_changed"] == (not allow_category)


def test_env_preserves_custom_acceptance_configuration(make_env):
    model = LogisticAcceptanceModel(1.0, {1: 100.0}, {1: 10.0})
    env = make_env(acceptance_model=model)
    assert env.acceptance_model is model
    assert model.high_season_multipliers == {}


@pytest.mark.parametrize("after_reset", [False, True])
def test_initial_bar_prices_are_positive_without_increments(make_env, monkeypatch, after_reset):
    env = make_env()
    if after_reset:
        env.bar_levels.fill(env._max_bar_level)
        env.reset(seed=42)

    np.testing.assert_array_equal(env.bar_levels, np.ones_like(env.bar_levels))
    assert env.observation_space.contains(env.get_observation())
    monkeypatch.setattr(env, "np_rng", Mock(random=lambda shape: np.ones(shape)))
    offers = []
    expected_prices = []
    for category in RoomCategory:
        for check_in in (env.start_simulation, env.extended_end_simulation - timedelta(days=2)):
            booking = Request(env.start_simulation, check_in, 2, 1)
            offers.append(HotelOffer(booking, category, check_in, 2))
            expected_prices.append([bar_table.get_price(category.name, 1)] * 2)

    prices = env._generate_prices(
        np.asarray(offers, dtype=object), np.zeros(env.action_space.shape, dtype=np.float32)
    )

    np.testing.assert_array_equal(prices, expected_prices)
    assert np.all(np.asarray(prices) > 0)
    np.testing.assert_array_equal(env.bar_levels, np.ones_like(env.bar_levels))


def test_bar_table_requires_initial_sellable_level(make_env, monkeypatch):
    monkeypatch.setattr(bar_table, "BAR_TABLE", {category: [0.0] for category in RoomCategory})
    with pytest.raises(ValueError, match="initial sellable level 1"):
        make_env()


@pytest.mark.parametrize("category_idx", range(3))
@pytest.mark.parametrize("season", range(2))
@pytest.mark.parametrize(
    "method", ["_update_bar_for_requests_vectorized", "_update_bar_for_requests"]
)
def test_six_bar_curves_are_independent(make_env, monkeypatch, category_idx, season, method):
    env = make_env()
    assert env.action_space.shape == (3, 2, 4)
    actions = np.zeros(env.action_space.shape, dtype=np.float32)
    actions[category_idx, season, 0] = 1
    probabilities = env._bar_increment_probabilities(actions)
    expected = np.zeros_like(probabilities)
    expected[category_idx, env.seasonality_array[1 : 1 + env.window_span_days] == season] = 0.25
    np.testing.assert_allclose(probabilities, expected)
    monkeypatch.setattr(env, "np_rng", Mock(random=lambda shape=None: np.full(shape or (), 0.1)))
    getattr(env, method)(actions)
    np.testing.assert_array_equal(env.bar_levels[:, 0], 1)
    np.testing.assert_array_equal(
        env.bar_levels[:, 1 : 1 + env.window_span_days], 1 + (expected > 0)
    )
    np.testing.assert_array_equal(env.bar_levels[:, 1 + env.window_span_days :], 1)
    env.bar_levels.fill(env._max_bar_level)
    getattr(env, method)(actions)
    np.testing.assert_array_equal(env.bar_levels, env._max_bar_level)


def test_seasonal_bar_prices_across_stay_boundary(make_env, monkeypatch):
    env = make_env()
    actions = np.zeros(env.action_space.shape, dtype=np.float32)
    actions[0, 1, 0] = 1
    monkeypatch.setattr(env, "np_rng", Mock(random=lambda shape: np.full(shape, 0.1)))
    booking = Request(env.today, date(2025, 6, 30), 2, 1)
    offer = HotelOffer(booking, RoomCategory.CATEGORY1, booking.t_reservation, 2)
    prices = env._generate_prices(np.asarray([offer], dtype=object), actions)
    assert prices == [[bar_table.get_price("CATEGORY1", 1), bar_table.get_price("CATEGORY1", 2)]]


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((3, 4)),
        np.full((3, 2, 4), np.nan),
        np.full((3, 2, 4), -0.1),
        np.full((3, 2, 4), 1.1),
    ],
)
def test_invalid_actions_do_not_mutate_env(make_env, actions):
    env = make_env()
    before = env.bar_levels.copy()
    today = env.today
    with pytest.raises(ValueError, match="Action"):
        env.step(actions)
    np.testing.assert_array_equal(env.bar_levels, before)
    assert env.today == today


def test_zero_capacity_category_has_no_bar_increments(make_env, monkeypatch):
    env = make_env()
    env.hotel = Hotel(0, 1, 1)
    monkeypatch.setattr(env, "np_rng", Mock(random=lambda shape: np.zeros(shape)))
    env._update_bar_for_requests_vectorized(np.ones(env.action_space.shape))
    np.testing.assert_array_equal(env.bar_levels[0], 1)
    np.testing.assert_array_equal(env.bar_levels[1:, 1 : 1 + env.window_span_days], 2)
    np.testing.assert_array_equal(env.bar_levels[:, 1 + env.window_span_days :], 1)
    env.occupancy[0, 1] = 1
    with pytest.raises(ValueError, match="without rooms"):
        env._bar_increment_probabilities(np.ones(env.action_space.shape))


def test_zero_actions_never_increment_bar(make_env, monkeypatch):
    env = make_env()
    monkeypatch.setattr(env, "np_rng", Mock(random=lambda shape: np.zeros(shape)))
    env.step(np.zeros(env.action_space.shape))
    np.testing.assert_array_equal(env.bar_levels, 1)


def test_scalar_and_vectorized_bar_updates_match(make_env):
    vectorized = make_env()
    scalar = make_env()
    occupancy = np.random.default_rng(7).integers(0, 2, size=vectorized.occupancy.shape)
    vectorized.occupancy[:] = occupancy
    scalar.occupancy[:] = occupancy
    actions = np.random.default_rng(8).uniform(size=vectorized.action_space.shape)
    vectorized._update_bar_for_requests_vectorized(actions)
    scalar._update_bar_for_requests(actions)
    np.testing.assert_array_equal(vectorized.bar_levels, scalar.bar_levels)


def test_seasonal_coefficients_and_calendar_are_saved_in_snapshot(make_env):
    env = make_env(collect_eval_artifacts=True, artifact_snapshot_dates=[date(2025, 1, 1)])
    actions = np.random.default_rng(8).uniform(size=env.action_space.shape)
    _, _, _, _, info = env.step(actions)
    snapshot = info["eval_artifacts"]
    np.testing.assert_array_equal(snapshot["policy_coefficients"], actions)
    np.testing.assert_array_equal(
        snapshot["seasonality_by_day"], env.seasonality_array[: env.total_days]
    )


def calendar_signature(generator):
    return [
        (day, [vars(request).copy() for request in requests])
        for day, requests in generator.requests_by_day.items()
    ]


def test_seed_replays_episode_despite_global_rng_changes(make_env):
    env = make_env()
    calendar = calendar_signature(env.request_generator)
    action = np.full(env.action_space.shape, 0.5, dtype=np.float32)

    def rollout():
        rewards = []
        observations = []
        for _ in range(env.total_days):
            observation, reward, _, _, _ = env.step(action)
            rewards.append(reward)
            observations.append(observation)
        return rewards, observations, env.occupancy.copy(), env.bar_levels.copy()

    env.reset(seed=123)
    first = rollout()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(999)
        np.random.seed(999)
        env.reset(seed=123)
        second = rollout()
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    assert first[0] == second[0]
    for first_obs, second_obs in zip(first[1], second[1], strict=True):
        for key in first_obs:
            np.testing.assert_array_equal(first_obs[key], second_obs[key])
    np.testing.assert_array_equal(first[2], second[2])
    np.testing.assert_array_equal(first[3], second[3])
    assert calendar_signature(env.request_generator) == calendar


def test_constructor_streams_are_separate_and_reset_keeps_progress(make_env):
    first, second = make_env(), make_env()
    assert first.acceptance_model.rng is first.acceptance_rng
    first.acceptance_rng.random(100)
    np.testing.assert_array_equal(first.np_rng.random(5), second.np_rng.random(5))
    np.testing.assert_array_equal(first.alternative_rng.random(5), second.alternative_rng.random(5))
    assert calendar_signature(first.request_generator) == calendar_signature(
        second.request_generator
    )
    streams = [
        first.np_rng,
        first.alternative_rng,
        first.acceptance_rng,
        first.request_generator.rng,
    ]
    states = [deepcopy(stream.bit_generator.state) for stream in streams]
    first.reset()
    assert [stream.bit_generator.state for stream in streams] == states
    assert first.np_rng is streams[0]
    assert first.alternative_rng is streams[1]
    assert first.acceptance_model.rng is streams[2]


@pytest.mark.parametrize("seed", [42, None])
def test_regeneration_advances_without_seed_and_replays_explicit_seed(make_env, seed):
    env = make_env(seed=seed)
    generator = env.request_generator
    state = deepcopy(generator.rng.bit_generator.state)
    env.reset(options={"new_requests": True})
    assert generator.rng.bit_generator.state != state
    env.reset(seed=123, options={"new_requests": True})
    expected = calendar_signature(generator)
    env.reset(options={"new_requests": True})
    assert calendar_signature(generator) != expected
    env.reset(seed=999, options={"new_requests": True, "requests_seed": 123})
    assert calendar_signature(generator) == expected


def test_generator_pickle_preserves_progress(make_env):
    generator = make_env().request_generator
    generator.regenerate()
    generator.py_rng.random()
    restored = pickle.loads(pickle.dumps(generator))
    assert calendar_signature(restored) == calendar_signature(generator)
    assert restored.py_rng.random() == generator.py_rng.random()
    generator.regenerate()
    restored.regenerate()
    assert calendar_signature(restored) == calendar_signature(generator)


def test_generator_pickle_uses_compact_records(make_env):
    generator = make_env().request_generator
    generator.requests_by_day[generator.end_simulation] = []
    state = generator.__getstate__()
    request_count = sum(len(requests) for requests in generator.requests_by_day.values())
    assert state["requests_format_version"] == 2
    assert (
        len(state["requests_by_day_bytes"])
        == 8 * len(generator.requests_by_day) + 12 * request_count
    )
    assert not hasattr(generator, "base_dist")
    restored = pickle.loads(pickle.dumps(generator))
    assert not hasattr(restored, "base_dist")
    assert calendar_signature(restored) == calendar_signature(generator)
    for requests in restored.requests_by_day.values():
        for request in requests:
            assert set(vars(request)) == {"t_request", "t_reservation", "days_stay", "num_guests"}


@pytest.mark.parametrize("with_rng_state", [True, False])
def test_generator_reads_unversioned_legacy_records(make_env, with_rng_state):
    generator = make_env().request_generator
    generator.requests_by_day[generator.end_simulation] = []
    generator.rng.random(10)
    generator.py_rng.random()
    state = generator.__getstate__()
    state.pop("requests_format_version")
    chunks = []
    for day, requests in generator.requests_by_day.items():
        chunks.append(struct.pack("<ii", day.toordinal(), len(requests)))
        for request in requests:
            chunks.append(
                struct.pack(
                    "<iihhih",
                    request.t_request.toordinal(),
                    request.t_reservation.toordinal(),
                    request.days_stay,
                    request.num_guests,
                    request.t_reservation.toordinal() + 1,
                    2,
                )
            )
    state["requests_by_day_bytes"] = b"".join(chunks)
    state["base_dist"] = "unused legacy distribution"
    if not with_rng_state:
        state.pop("rng")
        state.pop("py_rng")
    legacy = RequestGenerator.__new__(RequestGenerator)
    legacy.__setstate__(deepcopy(state))
    assert calendar_signature(legacy) == calendar_signature(generator)
    assert not hasattr(legacy, "base_dist")
    assert isinstance(legacy.rng, np.random.Generator)
    if with_rng_state:
        assert legacy.rng.random() == generator.rng.random()
        assert legacy.py_rng.random() == generator.py_rng.random()
    else:
        assert legacy.rng.random() == np.random.default_rng(generator.seed).random()
        assert legacy.py_rng.random() == random.Random(generator.seed).random()
    assert legacy.__getstate__()["requests_format_version"] == 2


def test_generator_rejects_unknown_pickle_version(make_env):
    state = make_env().request_generator.__getstate__()
    state["requests_format_version"] = 99
    restored = RequestGenerator.__new__(RequestGenerator)
    with pytest.raises(ValueError, match="Unsupported request serialization version"):
        restored.__setstate__(state)


def test_reset_clears_last_step_arrays(make_env):
    env = make_env()
    env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert env.last_total_requests.size > 0
    env.reset()
    for name in (
        "last_total_requests",
        "last_valid_requests",
        "last_valid_prices",
        "last_final_requests",
        "last_final_prices",
    ):
        assert getattr(env, name).size == 0
    assert env.last_day_avg_price == 0.0


@pytest.mark.parametrize("alternative", ["category", "period"])
@pytest.mark.parametrize("accepted", [True, False])
def test_simultaneous_offers_retry_capacity_conflict(make_env, monkeypatch, alternative, accepted):
    env = make_env(collect_eval_artifacts=True)
    category = 1 if alternative == "category" else 3
    booking = Request(env.today, env.today + timedelta(days=2), 1, category)
    env.request_generator.requests_by_day = {env.today: [booking, deepcopy(booking)]}
    calls = []

    def accept(total, offer):
        calls.append((total, offer, env.occupancy.copy()))
        return True if len(calls) <= 2 else accepted

    monkeypatch.setattr(env.acceptance_model, "accept", accept)
    update = Mock(wraps=env._update_bar_for_requests_vectorized)
    monkeypatch.setattr(env, "_update_bar_for_requests_vectorized", update)
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape))
    update.assert_called_once()
    assert len(calls) == 3
    assert not calls[0][2].any() and not calls[1][2].any()
    assert calls[2][2][category - 1, 2] == 1
    metrics = info["step_metrics"]
    assert metrics["num_requests"] == metrics["num_offers"] == 2
    assert metrics["num_confirmed"] == 1 + accepted
    assert reward == pytest.approx(calls[0][0] + (calls[2][0] if accepted else 0))
    assert env.occupancy.sum() == 1 + accepted
    rows = info["eval_artifacts"]["requests"]
    assert [len(row["offer_attempts"]) for row in rows] == [1, 2]
    assert [attempt["confirmed"] for attempt in rows[1]["offer_attempts"]] == [False, accepted]
    assert rows[1]["confirmed"] == accepted
    assert rows[1]["requested"]["category"] == category
    if alternative == "category":
        assert rows[1]["category"] == 2
    else:
        assert rows[1]["period_changed"]


def test_initial_price_refusal_is_not_retried(make_env, monkeypatch):
    env = make_env(collect_eval_artifacts=True)
    booking = Request(env.today, env.today + timedelta(days=1), 1, 1)
    env.request_generator.requests_by_day = {env.today: [booking]}
    accept = Mock(return_value=False)
    monkeypatch.setattr(env.acceptance_model, "accept", accept)
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape))
    accept.assert_called_once()
    assert reward == 0
    assert not env.occupancy.any()
    assert len(info["eval_artifacts"]["requests"][0]["offer_attempts"]) == 1


@pytest.mark.parametrize(
    "advance,nights,valid",
    [
        (0, 1, False),
        (1, 1, True),
        (360, 1, True),
        (359, 2, True),
        (360, 2, False),
        (361, 1, False),
    ],
)
def test_rolling_window_boundaries(make_env, advance, nights, valid):
    env = make_env()
    booking = Request(env.today, env.today + timedelta(days=advance), nights, 1)
    assert env.check_hotel_availability(booking) == valid
    mask, offers = env._process_requests(np.asarray([booking], dtype=object), env.today)
    assert mask.tolist() == [valid]
    assert len(offers) == int(valid)


def test_generator_can_book_last_visible_night(make_env, monkeypatch):
    env = make_env()
    rng = Mock()
    rng.choice.side_effect = lambda values, p: values[-1]
    rng.integers.side_effect = lambda low, high, **kwargs: low
    monkeypatch.setattr(env.request_generator, "rng", rng)
    booking = env.request_generator._generate_request(env.today)
    assert booking.advance == 360
    assert booking.days_stay == 1
    assert env._within_booking_window(booking)
    assert rng.integers.call_args_list[0].args == (1, 2)


def test_conflict_recovery_cannot_offer_night_outside_window(make_env, monkeypatch):
    env = make_env(collect_eval_artifacts=True)
    arrival = env.today + timedelta(days=360)
    booking = Request(env.today, arrival, 1, 3)
    env.request_generator.requests_by_day = {env.today: [booking, deepcopy(booking)]}
    env.occupancy[2, 359] = 1
    accept = Mock(return_value=True)
    monkeypatch.setattr(env.acceptance_model, "accept", accept)
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape))
    assert accept.call_count == 2
    assert info["step_metrics"]["num_confirmed"] == 1
    assert reward == bar_table.get_price("CATEGORY3", 1)
    assert info["eval_artifacts"]["bar_update_end_day"] == (arrival + timedelta(days=1)).isoformat()
    assert [len(row["offer_attempts"]) for row in info["eval_artifacts"]["requests"]] == [1, 1]


@pytest.mark.parametrize("alternative", ["category", "period"])
@pytest.mark.parametrize("accepted", [True, False])
def test_step_prices_and_books_the_offer_without_changing_demand(
    make_env, monkeypatch, alternative, accepted
):
    env = make_env(collect_eval_artifacts=True)
    category = 1 if alternative == "category" else 3
    request = Request(env.today, env.today + timedelta(days=2), 3, category)
    env.request_generator.requests_by_day = {env.today: [request, deepcopy(request)]}
    calendar = calendar_signature(env.request_generator)
    if alternative == "category":
        env.occupancy[0, 2:5] = 1
        env.occupancy[2, :] = 1
        expected_category, expected_start, expected_nights = 2, 2, 3
    else:
        env.occupancy[2, [2, 5]] = 1
        expected_category, expected_start, expected_nights = 3, 3, 2
    occupancy_before = env.occupancy.copy()
    env.bar_levels[:] = 1
    env.bar_levels[:, expected_start + 1] = 2
    monkeypatch.setattr(env.acceptance_model, "accept", lambda total, offer: accepted)

    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))

    expected_prices = [
        bar_table.get_price(
            RoomCategory(expected_category).name, int(env.bar_levels[expected_category - 1, index])
        )
        for index in range(expected_start, expected_start + expected_nights)
    ]
    rows = info["eval_artifacts"]["requests"]
    for row in rows:
        assert row["requested"]["check_in"] == request.t_reservation.isoformat()
        assert row["requested"]["days_stay"] == 3
        assert row["requested"]["category"] == category
        assert row["category"] == expected_category
        assert (
            row["check_in"] == (env.start_simulation + timedelta(days=expected_start)).isoformat()
        )
        assert row["days_stay"] == expected_nights
        assert row["num_guests"] == category
        assert row["period_changed"] == (alternative == "period")
        assert row["daily_prices"] == expected_prices
        assert row["price_accepted"] == accepted
    assert [row["confirmed"] for row in rows] == [accepted, False]
    expected_occupancy = occupancy_before.copy()
    if accepted:
        expected_occupancy[
            expected_category - 1, expected_start : expected_start + expected_nights
        ] += 1
    np.testing.assert_array_equal(env.occupancy, expected_occupancy)
    assert reward == pytest.approx(sum(expected_prices) if accepted else 0.0)
    assert env.demand[category - 1, 2] == 2
    assert env.demand.sum() == 2
    assert info["step_metrics"]["request_stays"] == [(2, 3), (2, 3)]
    assert calendar_signature(env.request_generator) == calendar

    env.reset()
    fresh = env._process_requests(
        env.request_generator.fetch_requests_for_date(env.today), env.today
    )[1]
    assert all(not offer.period_changed and offer.category.value == category for offer in fresh)
    assert rows[0]["category"] == expected_category
