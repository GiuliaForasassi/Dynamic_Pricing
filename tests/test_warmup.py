import pickle
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.plots.episode_metrics import EpisodeMetrics
from pricing_advertising.plots.plot_callback import PlotCallback
from pricing_advertising.pricing_model import bar_table
from pricing_advertising.requests.cancellation import CancellationModel
from pricing_advertising.requests.request import ALTERNATIVE_TYPES, OFFER_TYPES
from pricing_advertising.training.baseline_returns import evaluate_baseline_returns
from pricing_advertising.training.warmup_policy import (
    FrozenPolicy,
    WarmupPolicyCallback,
    sync_warmup_policy,
)
from pricing_advertising.utils.constants import NUM_WINDOWS, WINDOW_SIZE

START = date(2025, 1, 1)
END = date(2025, 1, 10)
WARMUP_DAYS = 20


@pytest.fixture(autouse=True)
def bar_table_loaded(monkeypatch):
    csv_path = Path(__file__).resolve().parents[1] / "data" / "bar_table.csv"
    monkeypatch.setattr(bar_table, "BAR_TABLE", bar_table.load_bar_table_wide_csv(csv_path))


def make_env(**kwargs) -> PricingEnv:
    config = {
        "start_simulation": START,
        "end_simulation": END,
        "hotel": Hotel(3, 3, 3),
        "avg_num_requests": 10,
        "seed": 42,
    }
    config.update(kwargs)
    return PricingEnv(**config)


def constant_policy(env: PricingEnv, value: float):
    return lambda _observation: np.full(env.action_space.shape, value, dtype=np.float32)


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_invalid_warmup_days_are_rejected(value):
    with pytest.raises(ValueError, match="warmup_days"):
        make_env(warmup_days=value)


def test_warmup_without_policy_fails_at_reset():
    env = make_env(warmup_days=WARMUP_DAYS)
    with pytest.raises(RuntimeError, match="warmup_policy"):
        env.reset()


def test_warmup_fills_the_hotel_without_changing_the_episode():
    env = make_env(warmup_days=WARMUP_DAYS)
    env.set_warmup_policy(constant_policy(env, 0.5))
    observation, _ = env.reset()

    assert env.today == START
    assert env.total_days == (END - START).days + 1
    assert env.occupancy[:, 0].sum() > 0
    assert observation["occupancy"].sum() > 0
    assert env.accepted_by_day.sum() == env.rejected_by_day.sum() == env.accepted_total == 0

    steps = 0
    terminated = False
    while not terminated:
        _, _, terminated, _, info = env.step(constant_policy(env, 0.5)(observation))
        assert info["step_metrics"]["day_index"] == steps
        steps += 1
    assert steps == env.total_days


@pytest.mark.parametrize("cancellation_model", [None, CancellationModel(daily_probability=0.2)])
def test_warmup_state_matches_an_environment_started_earlier(cancellation_model):
    env = make_env(warmup_days=WARMUP_DAYS, cancellation_model=cancellation_model)
    env.set_warmup_policy(constant_policy(env, 0.5))
    env.reset()

    earlier = make_env(
        start_simulation=START - timedelta(days=WARMUP_DAYS),
        request_generator=env.request_generator,
        seed=42,
        cancellation_model=cancellation_model,
    )
    earlier.reset()
    action = constant_policy(earlier, 0.5)(None)
    for _ in range(WARMUP_DAYS):
        earlier.step(action)

    kept = NUM_WINDOWS * WINDOW_SIZE
    assert env.occupancy[:, :kept].sum() > 0
    for name in ("occupancy", "demand", "bar_levels"):
        np.testing.assert_array_equal(
            getattr(env, name)[:, :kept], getattr(earlier, name)[:, WARMUP_DAYS:][:, :kept]
        )
    assert not env.occupancy[:, kept:].any()
    assert env.cancellation_rng.bit_generator.state == earlier.cancellation_rng.bit_generator.state


@pytest.mark.parametrize("cancellation_model", [None, CancellationModel(daily_probability=0.2)])
def test_stay_day_metrics_include_warmup_requests_and_bookings(monkeypatch, cancellation_model):
    env = make_env(warmup_days=WARMUP_DAYS, cancellation_model=cancellation_model)
    env.set_warmup_policy(constant_policy(env, 0.5))
    env.reset()
    earlier = make_env(
        start_simulation=START - timedelta(days=WARMUP_DAYS),
        request_generator=env.request_generator,
        seed=42,
        cancellation_model=cancellation_model,
    )
    earlier.reset()
    expected_requests = np.zeros(env.total_days, dtype=int)
    expected_revenue = np.zeros(env.total_days)
    expected_nights = np.zeros(env.total_days, dtype=int)
    warmup_requests = np.zeros(env.total_days, dtype=int)
    warmup_nights = np.zeros(env.total_days, dtype=int)
    expected_offers = np.zeros((env.total_days, len(OFFER_TYPES), 3))
    warmup_alternatives = 0
    relevant_cancellations = 0
    action = constant_policy(env, 0.5)(None)
    for day in range(WARMUP_DAYS + env.total_days):
        _, _, _, _, info = earlier.step(action)
        metrics = info["step_metrics"]
        for event in metrics["offer_stats"]["events"]:
            check_in = event["check_in_index"] - WARMUP_DAYS
            if 0 <= check_in < env.total_days:
                row = OFFER_TYPES.index(event["type"])
                expected_offers[check_in, row] += [1, event["price_accepted"], event["confirmed"]]
                if day < WARMUP_DAYS and event["type"] in ALTERNATIVE_TYPES:
                    warmup_alternatives += 1
        for cancellation in metrics["cancellations"]:
            offset = (date.fromisoformat(cancellation["check_in"]) - START).days
            for night, price in enumerate(cancellation["daily_prices"]):
                stay_day = offset + night
                if 0 <= stay_day < env.total_days:
                    expected_revenue[stay_day] -= price
                    expected_nights[stay_day] -= 1
                    relevant_cancellations += 1
        for check_in, nights in metrics["request_stays"]:
            for stay_day in range(check_in - WARMUP_DAYS, check_in - WARMUP_DAYS + nights):
                if 0 <= stay_day < env.total_days:
                    expected_requests[stay_day] += 1
                    if day < WARMUP_DAYS:
                        warmup_requests[stay_day] += 1
        prices = iter(metrics["confirmed_nightly_prices"])
        for offer in metrics["confirmed_requests"]:
            for offset in range(offer.days_stay):
                price = next(prices)
                stay_day = (offer.t_reservation - START).days + offset
                if 0 <= stay_day < env.total_days:
                    expected_revenue[stay_day] += price
                    expected_nights[stay_day] += 1
                    if day < WARMUP_DAYS:
                        warmup_nights[stay_day] += 1

    assert warmup_requests.sum() > 0
    assert warmup_nights.sum() > 0
    if cancellation_model is None:
        assert warmup_alternatives > 0
    else:
        assert relevant_cancellations > 0
    logged = {}
    monkeypatch.setattr("mlflow.log_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "mlflow.log_metric",
        lambda key, value, step: logged.setdefault(key, {}).update({step: value}),
    )
    tracker = EpisodeMetrics(split="val")
    for _ in range(env.total_days):
        _, _, _, _, info = env.step(action)
        tracker.record_day(info["step_metrics"])
    tracker.end_episode()
    tracker.finalize()

    request_values = logged["num_requests/by_stay_day/mean_completed_episodes/val"]
    price_values = logged["booked_mean_price/by_stay_day/pooled_completed_episodes/val"]
    np.testing.assert_array_equal(list(request_values.values()), expected_requests)
    np.testing.assert_array_equal(tracker.alternatives._by_checkin_sum, expected_offers)
    np.testing.assert_allclose(
        list(price_values.values()),
        np.divide(
            expected_revenue,
            expected_nights,
            out=np.full(env.total_days, np.nan),
            where=expected_nights > 0,
        ),
        equal_nan=True,
    )


@pytest.mark.parametrize("cancellation_model", [None, CancellationModel(daily_probability=0.2)])
def test_training_callback_counts_each_warmup_once_across_auto_resets(cancellation_model):
    def factory():
        env = make_env(warmup_days=WARMUP_DAYS, cancellation_model=cancellation_model)
        env.set_warmup_policy(constant_policy(env, 0.5))
        return env

    vec_env = DummyVecEnv([factory])
    callback = PlotCallback(log_to_mlflow=False)
    callback.model = SimpleNamespace(get_env=lambda: vec_env)
    reference = EpisodeMetrics(split="val", log_to_mlflow=False)
    vec_env.reset()
    action = np.full((1, *vec_env.action_space.shape), 0.5, dtype=np.float32)
    warmups_seen = 0
    for step in range(2 * (END - START).days + 2):
        _, _, dones, infos = vec_env.step(action)
        metrics = infos[0]["step_metrics"]
        if metrics["day_index"] == 0:
            warmups_seen += 1
            assert metrics["warmup_stay_metrics"]["requests"].sum() > 0
            assert np.all(metrics["warmup_stay_metrics"]["nights"] >= 0)
            if cancellation_model is None:
                assert metrics["warmup_stay_metrics"]["nights"].sum() > 0
        else:
            assert "warmup_stay_metrics" not in metrics
        reference.record_day(metrics)
        callback.locals = {"infos": infos, "dones": dones}
        callback.num_timesteps = step + 1
        callback._on_step()
        if dones[0]:
            reference.end_episode()

    assert warmups_seen == 2
    assert callback._request_tracker._completed_episodes == 2
    np.testing.assert_array_equal(
        callback._request_tracker._stays_sum, reference.requests._stays_sum
    )
    np.testing.assert_array_equal(
        callback._request_tracker._arrivals_sum, reference.requests._arrivals_sum
    )
    np.testing.assert_allclose(callback._price_tracker._revenue_sum, reference.price._revenue_sum)
    np.testing.assert_array_equal(callback._price_tracker._nights_sum, reference.price._nights_sum)
    np.testing.assert_array_equal(
        callback._alternatives_tracker._by_checkin_sum, reference.alternatives._by_checkin_sum
    )
    np.testing.assert_array_equal(
        callback._alternatives_tracker._totals_sum, reference.alternatives._totals_sum
    )
    vec_env.close()


def test_warmup_uses_the_policy_observations_with_zero_progress():
    env = make_env(warmup_days=3)
    seen = []

    def policy(observation):
        seen.append({key: np.copy(value) for key, value in observation.items()})
        return np.zeros(env.action_space.shape, dtype=np.float32)

    env.set_warmup_policy(policy)
    env.reset()
    assert len(seen) == 3
    assert all(observation["today"][0] == 0 for observation in seen)
    assert seen[0]["occupancy"].sum() == 0
    assert seen[-1]["occupancy"].sum() > 0


@pytest.mark.parametrize("mode", ["never", "always"])
def test_forced_bar_update_baselines_need_no_warmup_policy(mode):
    env = make_env(warmup_days=WARMUP_DAYS, bar_update_mode=mode)
    env.reset()
    assert env.occupancy[:, 0].sum() > 0


@pytest.mark.parametrize("cancellation_model", [None, CancellationModel(daily_probability=0.2)])
def test_reseeded_reset_replays_the_same_warmup(cancellation_model):
    env = make_env(
        warmup_days=WARMUP_DAYS, reseed_on_reset=True, cancellation_model=cancellation_model
    )
    env.set_warmup_policy(constant_policy(env, 0.5))
    env.reset()
    first = env.occupancy.copy(), env.bar_levels.copy()
    env.step(constant_policy(env, 0.5)(None))
    env.reset()
    np.testing.assert_array_equal(env.occupancy, first[0])
    np.testing.assert_array_equal(env.bar_levels, first[1])


def test_external_generator_must_cover_the_warmup_period():
    source = make_env()
    with pytest.raises(ValueError, match="cover"):
        make_env(warmup_days=WARMUP_DAYS, request_generator=source.request_generator)


def test_frozen_policy_is_a_picklable_snapshot_of_the_training_policy():
    env = DummyVecEnv([lambda: make_env()])
    model = PPO("MultiInputPolicy", env, n_steps=2, batch_size=2, seed=0)
    frozen = FrozenPolicy(model.policy, deterministic=True)
    observation = make_env().reset()[0]
    before = frozen(observation)

    for parameter in model.policy.parameters():
        parameter.data.add_(1.0)
    restored = pickle.loads(pickle.dumps(frozen))

    np.testing.assert_array_equal(restored(observation), before)
    assert before.shape == env.action_space.shape


def test_synced_policy_drives_the_warmup_of_training_envs():
    vec_env = DummyVecEnv([lambda: make_env(warmup_days=WARMUP_DAYS) for _ in range(2)])
    model = PPO("MultiInputPolicy", vec_env, n_steps=2, batch_size=2, seed=0)
    sync_warmup_policy(vec_env, model.policy, deterministic=False)
    observation = vec_env.reset()
    assert observation["occupancy"][:, :, 0].sum() > 0

    # Syncing again after an optimization step must still work
    model.learn(4, callback=WarmupPolicyCallback())
    model.learn(4, callback=WarmupPolicyCallback())
    assert all(isinstance(policy, FrozenPolicy) for policy in vec_env.get_attr("warmup_policy"))


def test_baseline_returns_run_with_warmup():
    returns = evaluate_baseline_returns(
        {
            "start_simulation": START,
            "end_simulation": END,
            "hotel": Hotel(3, 3, 3),
            "avg_num_requests": 10,
            "warmup_days": WARMUP_DAYS,
        },
        eval_seeds=[7],
        seed=1,
    )
    assert set(returns) == {"constant_bar", "always_increment", "random"}
    assert all(len(values) == 1 for values in returns.values())
