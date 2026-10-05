from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from gymnasium.error import ResetNeeded

from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.exceptions.exceptions import NotValidPeriodError
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.pricing_model import bar_table
from pricing_advertising.requests.request import HotelOffer, Request
from pricing_advertising.requests.request_generator import RequestGenerator
from pricing_advertising.utils.constants import NUM_WINDOWS, WINDOW_SIZE


@pytest.fixture
def make_env(monkeypatch):
    csv_path = Path(__file__).resolve().parents[1] / "data" / "bar_table.csv"
    monkeypatch.setattr(bar_table, "BAR_TABLE", bar_table.load_bar_table_wide_csv(csv_path))
    environments = []

    def factory(**kwargs):
        config = {
            "start_simulation": date(2025, 1, 1),
            "end_simulation": date(2025, 1, 2),
            "hotel": Hotel(1, 1, 1),
            "avg_num_requests": 0,
            "seed": 42,
        }
        config.update(kwargs)
        env = PricingEnv(**config)
        environments.append(env)
        return env

    yield factory
    for env in environments:
        env.close()


def test_external_generator_supplies_default_demand_normalization(make_env):
    source = make_env(avg_num_requests=5)
    env = make_env(avg_num_requests=None, request_generator=source.request_generator)
    assert env.avg_num_requests == 5
    env.demand[0, 1] = 5
    assert env.get_observation()["demand"][0, 0] == pytest.approx(1 / WINDOW_SIZE)
    assert env.request_generator is source.request_generator


def test_external_generator_allows_explicit_normalization(make_env):
    source = make_env(avg_num_requests=5)
    env = make_env(avg_num_requests=10, request_generator=source.request_generator)
    assert env.avg_num_requests == 10


@pytest.mark.parametrize("average", [-1, 1.5, float("nan"), float("inf"), True])
@pytest.mark.parametrize("external_generator", [False, True])
def test_invalid_average_is_rejected(make_env, average, external_generator):
    generator = make_env().request_generator if external_generator else None
    with pytest.raises(ValueError, match="non-negative integer"):
        make_env(avg_num_requests=average, request_generator=generator)


def test_empty_hotel_is_rejected_at_construction(make_env):
    with pytest.raises(ValueError, match="capacity must be positive"):
        make_env(hotel=Hotel(0, 0, 0))


@pytest.mark.parametrize("boundary", ["start", "end", "extended_end"])
def test_external_generator_must_cover_environment(make_env, boundary):
    env = make_env()
    generator = env.request_generator
    if boundary == "start":
        generator.start_simulation += timedelta(days=1)
    elif boundary == "end":
        generator.end_simulation -= timedelta(days=1)
    else:
        generator.extended_end_simulation -= timedelta(days=1)
    with pytest.raises(ValueError, match="cover"):
        make_env(request_generator=generator)


def test_external_generator_can_cover_a_larger_calendar(make_env):
    source = make_env(
        start_simulation=date(2024, 12, 31),
        end_simulation=date(2025, 1, 3),
    )
    env = make_env(request_generator=source.request_generator)
    assert env.observation_space.contains(env.reset()[0])


def test_external_generator_is_precomputed(make_env):
    env = make_env()
    generator = RequestGenerator(
        5,
        env.start_simulation,
        env.end_simulation,
        env.extended_end_simulation,
        env.seasonality_array,
        seed=42,
        precompute=False,
    )
    assert not generator.requests_by_day
    make_env(request_generator=generator)
    assert generator.requests_by_day


def test_step_after_termination_requires_reset_without_mutating_state(make_env):
    env = make_env(end_simulation=date(2025, 1, 1))
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, terminated, _, _ = env.step(action)
    assert terminated
    today = env.today
    arrays = [array.copy() for array in (env.occupancy, env.demand, env.bar_levels)]
    rng_state = deepcopy(env.np_rng.bit_generator.state)
    with pytest.raises(ResetNeeded, match="reset"):
        env.step(action)
    assert env.today == today
    assert env.np_rng.bit_generator.state == rng_state
    for actual, expected in zip((env.occupancy, env.demand, env.bar_levels), arrays, strict=True):
        np.testing.assert_array_equal(actual, expected)
    env.reset()
    assert env.step(action)[2]


@pytest.mark.parametrize("nights", [0, -1])
def test_quoting_invalid_stay_raises_instead_of_returning_empty_price(make_env, nights):
    env = make_env()
    request = Request(env.today, env.today + timedelta(days=1), nights, 1)
    offer = HotelOffer(request, request.category, request.t_reservation, nights)
    with pytest.raises(NotValidPeriodError):
        env._quote_prices(np.asarray([offer], dtype=object))


def test_observations_preserve_float32_bounds_and_do_not_alias_state(make_env):
    env = make_env(hotel=Hotel(0, 2, 3), avg_num_requests=5)
    env.occupancy[1, 1:] = 2
    env.demand[1, 1:] = 1000
    env.bar_levels[1, 1:] = env._max_bar_level
    observation = env.get_observation()
    assert env.observation_space.contains(observation)
    for key in ("occupancy", "demand", "bar_levels"):
        assert observation[key].dtype == np.float32
        assert observation[key].shape == (3, NUM_WINDOWS)
        np.testing.assert_array_equal(observation[key][1], 1)
        assert not np.shares_memory(observation[key], getattr(env, key))
    np.testing.assert_array_equal(observation["occupancy"][0], 0)
    env.today = env.end_simulation + timedelta(days=1)
    assert env.observation_space.contains(env.get_observation())


def test_empty_price_evaluation_returns_boolean_mask_and_float_prices(make_env):
    env = make_env()
    accepted, prices = env._evaluate_prices([], np.array([], dtype=object))
    assert accepted.shape == prices.shape == (0,)
    assert accepted.dtype == np.bool_
    assert prices.dtype == np.float64
