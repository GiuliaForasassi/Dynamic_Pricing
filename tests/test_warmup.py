import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.pricing_model import bar_table
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


def test_warmup_state_matches_an_environment_started_earlier():
    env = make_env(warmup_days=WARMUP_DAYS)
    env.set_warmup_policy(constant_policy(env, 0.5))
    env.reset()

    earlier = make_env(
        start_simulation=START - timedelta(days=WARMUP_DAYS),
        request_generator=env.request_generator,
        seed=42,
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


def test_reseeded_reset_replays_the_same_warmup():
    env = make_env(warmup_days=WARMUP_DAYS, reseed_on_reset=True)
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
