import csv
from copy import deepcopy
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import TypedDict
from unittest.mock import MagicMock

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.plots import plot_callback as module
from pricing_advertising.plots.eval_artifact_plots import build_snapshot_metrics
from pricing_advertising.pricing_model import bar_table
from pricing_advertising.requests.request import Request

START = date(2025, 12, 30)
END = date(2025, 12, 31)


class PPOOverrides(TypedDict, total=False):
    run_name: str
    batch_size: int
    ent_coef: float
    gae_lambda: float
    clip_range: float
    net_arch: list[int] | dict[str, list[int]]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"eval_frequency": 0}, "eval_frequency must be a positive integer"),
        ({"eval_frequency": -1}, "eval_frequency must be a positive integer"),
        ({"eval_frequency": True}, "eval_frequency must be a positive integer"),
        ({"eval_frequency": 1.5}, "eval_frequency must be a positive integer"),
        ({"eval_seeds": []}, "At least one evaluation seed"),
        ({"eval_seeds": [100, 100]}, "Evaluation seeds must be distinct"),
        ({"eval_seeds": [43]}, "distinct from the training seeds"),
        ({"eval_seeds": [2**32]}, "Seeds must belong to the interval"),
    ],
)
def test_invalid_ppo_evaluation_configuration_fails_before_starting_run(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, int | float | list[int]],
    message: str,
) -> None:
    from pricing_advertising.training import ppo_training as training_module

    mlflow = MagicMock()
    monkeypatch.setattr(training_module, "mlflow", mlflow)
    parameters = dict(
        total_timesteps=4,
        n_envs=2,
        n_steps=2,
        learning_rate=3e-4,
        gamma=0.995,
        n_epochs=1,
        policy="MultiInputPolicy",
        start_simulation=START,
        end_simulation=END,
        num_category1_rooms=2,
        num_category2_rooms=2,
        num_category3_rooms=2,
        seed=42,
        eval_seeds=[100, 101],
        eval_frequency=2,
    )
    parameters.update(overrides)
    with pytest.raises(ValueError, match=message):
        training_module.run_simulation(**parameters)
    mlflow.start_run.assert_not_called()


@pytest.mark.parametrize("eval_frequency", [2, 3])
@pytest.mark.parametrize("n_envs", [1, 2])
@pytest.mark.parametrize("allow_category", [False, True])
@pytest.mark.parametrize("allow_period", [False, True])
@pytest.mark.parametrize(
    "ppo_kwargs",
    [
        {},
        dict(
            run_name="PPO_custom_test",
            batch_size=2,
            ent_coef=0.01,
            gae_lambda=0.9,
            clip_range=0.1,
            net_arch=[16, 16],
        ),
        dict(
            batch_size=2,
            ent_coef=0.005,
            gae_lambda=0.98,
            clip_range=0.3,
            net_arch=dict(pi=[16, 8], vf=[32, 16]),
        ),
    ],
    ids=["ppo-defaults", "symmetric-network", "separate-networks"],
)
def test_training_run_saves_vecnormalize_and_evaluates_in_euros(
    make_env,
    monkeypatch,
    tmp_path,
    allow_category,
    allow_period,
    ppo_kwargs: PPOOverrides,
    n_envs: int,
    eval_frequency: int,
):
    from pricing_advertising.training import ppo_training as training_module

    monkeypatch.chdir(tmp_path)
    mlflow = MagicMock()
    mlflow.start_run.return_value.__enter__.return_value.info.run_id = "normalization_test"
    monkeypatch.setattr(training_module, "mlflow", mlflow)
    monkeypatch.setattr(training_module, "MLflowOutputFormat", lambda: MagicMock())
    normalizers = []
    callbacks = []
    eval_seeds = [100, 101, 102]

    def make_normalizer(env, **kwargs):
        normalizer = VecNormalize(env, **kwargs)
        normalizers.append(normalizer)
        return normalizer

    def make_vector(factories, start_method):
        assert start_method == "fork"
        for factory in factories:
            assert factory.keywords["allow_category_alternative"] is allow_category
            assert factory.keywords["allow_period_alternative"] is allow_period
        if normalizers:
            assert [factory.keywords["seed"] for factory in factories] == eval_seeds
            assert all(factory.keywords["reseed_on_reset"] for factory in factories)
        price = 100.0 if not normalizers else 200.0
        return DummyVecEnv([lambda: make_env(price=price)[0] for _ in factories])

    def make_evaluation(*args, **kwargs):
        kwargs["log_to_mlflow"] = False
        callback = module.EvalOccupancyCallback(*args, **kwargs)
        callbacks.append(callback)
        return callback

    monkeypatch.setattr(training_module, "VecNormalize", make_normalizer)
    monkeypatch.setattr(training_module, "SubprocVecEnv", make_vector)
    monkeypatch.setattr(
        training_module, "PlotCallback", lambda **kwargs: module.PlotCallback(log_to_mlflow=False)
    )
    monkeypatch.setattr(training_module, "EvalOccupancyCallback", make_evaluation)
    mean_reward = training_module.run_simulation(
        total_timesteps=4 * n_envs,
        n_envs=n_envs,
        n_steps=2,
        learning_rate=3e-4,
        gamma=0.995,
        n_epochs=1,
        policy="MultiInputPolicy",
        start_simulation=START,
        end_simulation=END,
        num_category1_rooms=2,
        num_category2_rooms=2,
        num_category3_rooms=2,
        seed=42,
        eval_seeds=eval_seeds,
        eval_frequency=eval_frequency,
        avg_num_requests=5,
        allow_category_alternative=allow_category,
        allow_period_alternative=allow_period,
        **ppo_kwargs,
    )
    assert mean_reward == 600.0
    mlflow.start_run.assert_called_once_with(
        run_name=ppo_kwargs.get("run_name", "PPO_vec_normalize"), nested=True
    )
    training, evaluation = normalizers
    assert training.num_envs == n_envs
    assert evaluation.num_envs == len(eval_seeds)
    assert callbacks[0].n_eval_episodes == len(eval_seeds)
    assert callbacks[0].eval_freq == eval_frequency
    effective_eval_frequency = eval_frequency * n_envs
    assert callbacks[0].evaluations_timesteps == list(
        range(effective_eval_frequency, 4 * n_envs + 1, effective_eval_frequency)
    )
    assert all(len(returns) == len(eval_seeds) for returns in callbacks[0].evaluations_results)
    logged_params = {
        key: value
        for call in mlflow.log_params.call_args_list
        for key, value in call.args[0].items()
    }
    assert logged_params["n_eval_envs"] == logged_params["eval_episodes"] == len(eval_seeds)
    assert logged_params["eval_frequency"] == eval_frequency
    assert logged_params["effective_eval_frequency"] == effective_eval_frequency
    assert training.gamma == evaluation.gamma == 0.995
    assert training.norm_obs is evaluation.norm_obs is False
    assert training.norm_reward is True
    assert evaluation.norm_reward is evaluation.training is False
    for normalizer in normalizers:
        calendar_space = normalizer.observation_space["calendar"]
        assert calendar_space.shape == (2,)
        np.testing.assert_array_equal(calendar_space.low, [-1, -1])
        np.testing.assert_array_equal(calendar_space.high, [1, 1])
        original = normalizer.get_original_obs()
        np.testing.assert_array_equal(
            normalizer.normalize_obs(original)["calendar"], original["calendar"]
        )
    output = tmp_path / "models" / "vec_normalize" / "normalization_test"
    for filename in (
        "final_model.zip",
        "final_vecnormalize.pkl",
        "best_model/best_model.zip",
        "best_model/best_vecnormalize.pkl",
    ):
        assert (output / filename).is_file()
    restored_model = PPO.load(output / "final_model.zip", device="cpu")
    assert restored_model.batch_size == ppo_kwargs.get("batch_size", 64)
    assert restored_model.ent_coef == ppo_kwargs.get("ent_coef", 0.0)
    assert restored_model.gae_lambda == ppo_kwargs.get("gae_lambda", 0.95)
    assert restored_model.clip_range(1.0) == ppo_kwargs.get("clip_range", 0.2)
    assert restored_model.policy.net_arch == ppo_kwargs.get(
        "net_arch", dict(pi=[64, 64], vf=[64, 64])
    )
    assert restored_model.observation_space["calendar"].shape == (2,)
    actions, _ = restored_model.predict(training.get_original_obs(), deterministic=True)
    assert actions.shape == (n_envs, 3, 2, 4)
    mlflow.log_metric.assert_any_call("final_eval_mean_reward", 600.0)
    logged_params = {}
    for call in mlflow.log_params.call_args_list:
        logged_params.update(call.args[0])
    assert logged_params["allow_category_alternative"] is allow_category
    assert logged_params["allow_period_alternative"] is allow_period
    for name, default in (
        ("batch_size", 64),
        ("ent_coef", 0.0),
        ("gae_lambda", 0.95),
        ("clip_range", 0.2),
    ):
        assert logged_params[name] == ppo_kwargs.get(name, default)
    assert logged_params["net_arch"] == str(restored_model.policy.net_arch)


def test_vecnormalize_preserves_euro_metrics_and_best_checkpoint(
    make_env, logged, monkeypatch, tmp_path
):
    monkeypatch.setattr(module, "log_eval_artifact_plots", lambda artifact: None)
    training_env, _ = make_env(price=100.0)
    evaluation_env, _ = make_env(price=200.0, collect_artifacts=True)
    training = VecNormalize(
        VecMonitor(DummyVecEnv([lambda: training_env])),
        norm_obs=False,
        norm_reward=True,
        gamma=0.995,
    )
    evaluation = VecNormalize(
        VecMonitor(DummyVecEnv([lambda: evaluation_env])),
        training=False,
        norm_obs=False,
        norm_reward=False,
        gamma=0.995,
    )
    model = PPO(
        "MultiInputPolicy",
        training,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        gamma=0.995,
        seed=42,
        device="cpu",
    )
    model.set_logger(Logger(folder=None, output_formats=[]))
    callback = module.EvalOccupancyCallback(
        evaluation,
        eval_freq=2,
        n_eval_episodes=1,
        best_model_save_path=str(tmp_path),
        verbose=0,
    )
    try:
        observation = training.reset()
        for key, value in training_env.get_observation().items():
            np.testing.assert_array_equal(observation[key][0], value)
        _, rewards, _, infos = training.step(np.zeros((1, *training.action_space.shape)))
        assert rewards[0] != 100.0
        assert abs(rewards[0]) <= training.clip_reward
        assert infos[0]["step_metrics"]["reward"] == 100.0
        model.learn(total_timesteps=4, callback=callback)
        assert callback.best_mean_reward == 600.0
        assert (tmp_path / "best_model.zip").is_file()
        assert (tmp_path / "best_vecnormalize.pkl").is_file()
        assert evaluation.training is False
        assert evaluation.norm_reward is False
        assert evaluation.ret_rms.count == training.ret_rms.count
        np.testing.assert_allclose(evaluation.ret_rms.var, training.ret_rms.var)
        count_before = evaluation.ret_rms.count
        evaluation.reset()
        _, rewards, _, _ = evaluation.step(np.zeros((1, *evaluation.action_space.shape)))
        assert rewards[0] == 200.0
        assert evaluation.ret_rms.count == count_before
        final_path = tmp_path / "final_vecnormalize.pkl"
        training.save(str(final_path))
        for filename in ("best_vecnormalize.pkl", "final_vecnormalize.pkl"):
            restored_env, _ = make_env()
            restored = VecNormalize.load(
                str(tmp_path / filename), VecMonitor(DummyVecEnv([lambda: restored_env]))
            )
            try:
                assert restored.gamma == 0.995
                assert restored.norm_obs is False
                assert restored.norm_reward is True
                if filename == "best_vecnormalize.pkl":
                    assert restored.ret_rms.count < training.ret_rms.count
                else:
                    np.testing.assert_allclose(restored.ret_rms.var, training.ret_rms.var)
                    np.testing.assert_allclose(restored.ret_rms.mean, training.ret_rms.mean)
                    assert restored.ret_rms.count == training.ret_rms.count
                previous_count = restored.ret_rms.count
                restored.reset()
                restored.step(np.zeros((1, *restored.action_space.shape)))
                assert restored.ret_rms.count > previous_count
            finally:
                restored.close()
        metrics, _, _ = logged
        assert points(metrics, "episode_return", "val") == [(2, 600.0), (4, 600.0)]
    finally:
        training.close()
        evaluation.close()


class ZeroPolicy:
    """Sostituisce PPO, non l'ambiente o i callback di evaluation."""

    def __init__(self, env):
        self.env = env
        self.num_timesteps = 0
        self.logger = Logger(folder=None, output_formats=[])

    def get_env(self):
        return self.env

    def get_vec_normalize_env(self):
        return None

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic=True,
    ):
        actions = np.zeros(
            (self.env.num_envs, *self.env.action_space.shape),
            dtype=np.float32,
        )
        return actions, state


@pytest.fixture
def logged(monkeypatch):
    metrics = []
    texts = {}
    artifacts = {}

    def log_metric(key, value, step):
        metrics.append((key, float(value), step))

    def log_text(text, artifact_file):
        assert artifact_file not in texts, "CSV sovrascritto"
        texts[artifact_file] = text

    def log_dict(dictionary, artifact_file):
        assert artifact_file not in artifacts, "Artifact sovrascritto"
        artifacts[artifact_file] = deepcopy(dictionary)

    monkeypatch.setattr(module.mlflow, "log_metric", log_metric)
    monkeypatch.setattr(module.mlflow, "log_text", log_text)
    monkeypatch.setattr(module.mlflow, "log_dict", log_dict)

    return metrics, texts, artifacts


@pytest.fixture(autouse=True)
def no_figure_upload(monkeypatch):
    monkeypatch.setattr(module.mlflow, "log_figure", lambda fig, path: None)


@pytest.fixture
def make_env(monkeypatch):
    csv_path = Path(__file__).resolve().parents[1] / "data" / "bar_table.csv"
    monkeypatch.setattr(
        bar_table,
        "BAR_TABLE",
        bar_table.load_bar_table_wide_csv(csv_path),
    )

    def factory(price=100.0, collect_artifacts=False, start=START, end=END):
        env = PricingEnv(
            start_simulation=start,
            end_simulation=end,
            hotel=Hotel(
                num_category1_rooms=2,
                num_category2_rooms=2,
                num_category3_rooms=2,
            ),
            avg_num_requests=5,
            seed=42,
            collect_eval_artifacts=collect_artifacts,
            # Testiamo i dati JSON, senza generare immagini.
            artifact_snapshot_dates=[],
        )

        env.request_generator.requests_by_day = {
            start: [Request(start, end, 1, 1)],
            end: [Request(end, end + timedelta(days=2), 2, 1)],
        }

        pricing = {"nightly": float(price)}

        monkeypatch.setattr(
            env,
            "_generate_prices",
            lambda requests, action: [
                [pricing["nightly"]] * int(request.days_stay) for request in requests
            ],
        )
        monkeypatch.setattr(
            env,
            "_evaluate_prices",
            lambda prices, requests: (
                np.ones(len(requests), dtype=bool),
                np.asarray([sum(values) for values in prices], dtype=float),
            ),
        )

        return env, pricing

    return factory


@pytest.mark.parametrize("with_booking", [False, True])
def test_daily_bar_history_tracks_updates_and_booking_order(make_env, monkeypatch, with_booking):
    env, _ = make_env(collect_artifacts=True)
    env.artifact_snapshot_dates = {START, END}
    monkeypatch.setattr(env, "_generate_prices", PricingEnv._generate_prices.__get__(env))
    monkeypatch.setattr(env, "np_rng", MagicMock(random=lambda shape: np.zeros(shape)))
    if not with_booking:
        env.request_generator.requests_by_day[START] = []
    env.bar_levels.fill(env._max_bar_level - 1)
    action = np.ones(env.action_space.shape, dtype=np.float32)

    try:
        _, _, _, _, info = env.step(action)
        first_payload = info["eval_artifacts"]
        original = deepcopy(first_payload)
        point = first_payload["bar_history"][END.isoformat()]
        assert point["levels_before"] == [env._max_bar_level - 1] * 3
        assert point["levels_after"] == [env._max_bar_level] * 3
        assert point["occupancy_before"] == [0, 0, 0]
        assert point["occupancy_after"] == [int(with_booking), 0, 0]
        assert (
            first_payload["bar_history"][START.isoformat()]["levels_after"]
            == [env._max_bar_level - 1] * 3
        )

        _, _, terminated, _, info = env.step(action)
        assert terminated
        assert list(info["eval_artifacts"]["bar_history"]) == [END.isoformat()]
        last_point = info["eval_artifacts"]["bar_history"][END.isoformat()]
        assert last_point["levels_before"] == last_point["levels_after"]
        assert last_point["levels_after"] == [env._max_bar_level] * 3
        assert last_point["occupancy_before"] == point["occupancy_after"]
        env.reset()
        assert first_payload == original
    finally:
        env.close()


def points(metrics, name, split):
    key = f"{name}/by_training_timesteps/{split}"
    return [(step, value) for logged_key, value, step in metrics if logged_key == key]


def assert_temporal_metrics(metrics, split, checkpoints):
    """checkpoints contiene coppie (timestep, prezzo notturno)."""
    expected = {
        "num_requests_per_episode": lambda price: 2.0,
        "episode_return": lambda price: 3.0 * price,
        "mean_daily_reward": lambda price: 1.5 * price,
        "occupancy_pct": lambda price: 100.0 / 12,
        "booked_mean_price": lambda price: price,
        "confirmation_rate": lambda price: 1.0,
        "price_acceptance_rate": lambda price: 1.0,
    }

    for name, compute in expected.items():
        actual = points(metrics, name, split)
        assert [step for step, _ in actual] == [step for step, _ in checkpoints], name
        assert [value for _, value in actual] == pytest.approx(
            [compute(price) for _, price in checkpoints]
        ), name


def test_train_callback_keeps_terminal_info_and_ignores_partial_episode(
    make_env,
    logged,
):
    metrics, texts, artifacts = logged
    env0, _ = make_env(price=100.0)
    env1, _ = make_env(price=900.0)

    # Il secondo ambiente ha prezzi diversi:
    # rileviamo se il callback mescola accidentalmente i due worker.
    vec = VecMonitor(DummyVecEnv([lambda: env0, lambda: env1]))
    model = ZeroPolicy(vec)
    callback = module.PlotCallback(track_env_idx=0)
    callback.init_callback(model)
    callback.on_training_start({}, {})

    try:
        vec.reset()
        actions = np.zeros(
            (2, *vec.action_space.shape),
            dtype=np.float32,
        )

        # Un episodio completo e un giorno del secondo episodio.
        for index in range(3):
            _, _, dones, infos = vec.step(actions)
            model.num_timesteps += vec.num_envs

            if index == 1:
                assert dones[0]
                assert env0.today == START
                assert not env0.occupancy.any()

                terminal = infos[0]["step_metrics"]
                assert terminal["day"] == END
                assert terminal["reward"] == 200.0
                assert terminal["occupancy_pct"] == pytest.approx(100 / 6)

            callback.update_locals({"infos": infos, "dones": dones})
            assert callback.on_step()

        callback.on_training_end()

        # Due giorni * due ambienti = timestep 4.
        # Il giorno parziale successivo non produce un altro punto.
        assert_temporal_metrics(metrics, "train", [(4, 100.0)])

        reward_rows = list(
            csv.DictReader(StringIO(texts["episodes/train/episode_00000_reward.csv"]))
        )
        assert [int(row["day"]) for row in reward_rows] == [1, 2]
        assert [float(row["reward"]) for row in reward_rows] == [100, 200]

        price_rows = list(csv.DictReader(StringIO(texts["episodes/train/episode_00000_price.csv"])))
        occupancy_rows = list(
            csv.DictReader(StringIO(texts["episodes/train/episode_00000_occupancy.csv"]))
        )

        assert [int(row["room_nights"]) for row in price_rows] == [0, 1]
        for price_row, occupancy_row in zip(price_rows, occupancy_rows, strict=True):
            assert price_row["day"] == occupancy_row["day"]
            assert float(occupancy_row["occupancy_pct"]) == pytest.approx(
                100 * int(price_row["room_nights"]) / 6
            )

        daily_reward = [
            (step, value)
            for key, value, step in metrics
            if key == "reward/by_request_day/mean_completed_episodes/train"
        ]
        assert daily_reward == [(1, 100.0), (2, 200.0)]

        assert len(texts) == 4  # Richieste, reward, occupancy, prezzi.
        assert all("episode_00000_" in path for path in texts)
        assert not [path for path in artifacts if not path.startswith("aggregation_data/")]
    finally:
        vec.close()


def test_evaluation_with_several_envs_keeps_episodes_separate(make_env, logged):
    metrics, texts, artifacts = logged
    training_env, _ = make_env()
    eval_envs = [make_env(price=price, collect_artifacts=True)[0] for price in (100.0, 300.0)]
    training = VecMonitor(DummyVecEnv([lambda: training_env]))
    evaluation = VecMonitor(DummyVecEnv([lambda env=env: env for env in eval_envs]))
    model = ZeroPolicy(evaluation)
    callback = module.EvalOccupancyCallback(
        evaluation, n_eval_episodes=2, eval_freq=1, best_model_save_path=None, log_path=None
    )
    callback.init_callback(model)
    callback.on_training_start({}, {})

    try:
        training.reset()
        _, _, _, infos = training.step(np.zeros((1, *training.action_space.shape)))
        callback.update_locals({"infos": infos})
        model.num_timesteps = 2
        assert callback.on_step()
        callback.on_training_end()
    finally:
        training.close()
        evaluation.close()

    for env_idx in range(2):
        assert f"eval_data/00000002_timesteps/env_{env_idx:02d}/episode_00000.json" in artifacts
    rewards = [
        [float(row["reward"]) for row in csv.DictReader(StringIO(texts[path]))]
        for path in (
            "episodes/val/episode_00000_reward.csv",
            "episodes/val/episode_00001_reward.csv",
        )
    ]
    assert rewards == [[100.0, 200.0], [300.0, 600.0]]
    returns = {value for key, value, _ in metrics if key.startswith("episode_return/")}
    assert returns == {600.0}  # mean of the two episodes: (300 + 900) / 2


def test_artifact_figures_use_only_snapshot_requests_from_evaluation(make_env, logged, monkeypatch):
    _, _, artifacts = logged
    training_env, _ = make_env(price=900.0)
    stay_day = END + timedelta(days=1)
    evaluation_env, _ = make_env(collect_artifacts=True, end=stay_day)
    evaluation_env.artifact_snapshot_dates = {END}
    evaluation_env.request_generator.requests_by_day = {
        START: [Request(START, stay_day, 1, 2)],
        END: [Request(END, stay_day, 1, 1), Request(END, stay_day, 1, 1)],
    }
    monkeypatch.setattr(
        evaluation_env,
        "_generate_prices",
        lambda requests, action: [
            [100.0 if evaluation_env.today == START else 200.0] * int(request.days_stay)
            for request in requests
        ],
    )
    figures = {}
    monkeypatch.setattr(module.mlflow, "log_figure", lambda fig, path: figures.update({path: fig}))
    training = VecMonitor(DummyVecEnv([lambda: training_env]))
    evaluation = VecMonitor(DummyVecEnv([lambda: evaluation_env]))
    model = ZeroPolicy(training)
    callback = module.EvalOccupancyCallback(
        evaluation,
        n_eval_episodes=2,
        eval_freq=1,
        best_model_save_path=None,
        log_path=None,
        verbose=0,
    )
    callback.init_callback(model)
    callback.on_training_start({}, {})

    try:
        training.reset()
        _, _, _, training_infos = training.step(np.zeros((1, *training.action_space.shape)))
        assert training_infos[0]["step_metrics"]["reward"] == 900.0
        assert "eval_artifacts" not in training_infos[0]
        callback.update_locals({"infos": training_infos})
        model.num_timesteps = 2
        assert callback.on_step()
        assert len(artifacts) == 2
        assert len(figures) == 16
        assert not any(path.startswith("request_plots/") for path in figures)

        for episode in range(2):
            suffix = f"00000002_timesteps/env_00/episode_{episode:05d}"
            artifact = artifacts[f"eval_data/{suffix}.json"]
            snapshot = artifact["steps"][1]
            assert snapshot["request_day"] == str(END)
            assert snapshot["occupancy_by_category"] == [[0, 2], [0, 1], [0, 0]]
            assert artifact["steps"][0]["requests"][0]["daily_prices"] == [100.0]
            assert [row["daily_prices"] for row in snapshot["requests"]] == [[200.0], [200.0]]
            for folder in ("proposed_price_by_advance", "booked_price_plots"):
                fig = figures[f"{folder}/{END}/{suffix}.png"]
                assert len(fig.axes) == 3
                assert all(len(panel.lines) == 1 for panel in fig.axes)
                expected_x = [1] if folder == "proposed_price_by_advance" else [3]
                np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), expected_x)
                np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [200])
                assert all(panel.lines[0].get_ydata().size == 0 for panel in fig.axes[1:])
                if folder == "proposed_price_by_advance":
                    np.testing.assert_array_equal(
                        fig.axes[0].collections[0].get_offsets(), [[1, 200], [1, 200]]
                    )
                    assert all(not panel.collections for panel in fig.axes[1:])
                else:
                    assert all(not panel.collections for panel in fig.axes)
            for folder in ("confirmation_rate_plots", "price_acceptance_plots"):
                fig = figures[f"{folder}/{END}/{suffix}.png"]
                np.testing.assert_array_equal(fig.axes[0].collections[0].get_offsets(), [[3, 1]])
            for folder, expected in (("occupancy_plots", [0, 100 * 2 / 6]),):
                fig = figures[f"{folder}/{END}/{suffix}.png"]
                np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), expected)
            for folder in ("bar_by_advance",):
                fig = figures[f"{folder}/{END}/{suffix}.png"]
                for category_idx in range(3):
                    before = np.asarray(snapshot["bar_levels_before_update"][category_idx])
                    after = np.asarray(snapshot["bar_levels_by_category"][category_idx])
                    increased_advance = np.flatnonzero(after > before)
                    np.testing.assert_array_equal(
                        fig.axes[1].collections[category_idx].get_offsets(),
                        np.column_stack(
                            (
                                increased_advance,
                                np.full(len(increased_advance), category_idx + 1),
                            )
                        ),
                    )
                    np.testing.assert_array_equal(
                        fig.axes[0].lines[category_idx].get_ydata(),
                        snapshot["bar_levels_by_category"][category_idx],
                    )
            fig = figures[f"bar_price_plots/{END}/{suffix}.png"]
            for category_idx, line in enumerate(fig.axes[0].lines):
                np.testing.assert_array_equal(
                    line.get_ydata(), snapshot["bar_prices_by_category"][category_idx]
                )
            fig = figures[f"policy_diagnostics/{END}/{suffix}.png"]
            for category_idx, panel in enumerate(fig.axes[::3]):
                assert len(panel.containers) == 2
                for season, container in enumerate(panel.containers):
                    np.testing.assert_array_equal(
                        [bar.get_height() for bar in container],
                        snapshot["policy_coefficients"][category_idx][season],
                    )
        for path, fig in figures.items():
            for panel in fig.axes:
                if path.startswith("policy_diagnostics/") and any(
                    panel is axis for axis in fig.axes[::3]
                ):
                    assert panel.containers
                    assert not panel.lines
                    continue
                if path.startswith("bar_by_advance/") and (panel is fig.axes[1]):
                    assert not panel.lines
                    assert len(panel.collections) == 3
                elif path.startswith(("confirmation_rate_plots/", "price_acceptance_plots/")):
                    assert not panel.lines
                    assert len(panel.collections) == 1
                else:
                    assert panel.lines
                    if not path.startswith("proposed_price_by_advance/"):
                        assert not panel.collections
                assert not panel.images
                assert not panel.containers
                assert all(line.get_linestyle() != "None" for line in panel.lines)
        callback.on_training_end()
    finally:
        training.close()
        evaluation.close()


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("end", [date(2025, 3, 1), date(2025, 3, 11)])
def test_reward_is_recognized_on_booking_day_in_metrics_and_artifacts(
    make_env, logged, monkeypatch, split, end
):
    metrics, texts, artifacts = logged
    booking_day = date(2025, 2, 28)
    env, _ = make_env(collect_artifacts=True, start=booking_day, end=end)
    env.artifact_snapshot_dates = {booking_day}
    env.request_generator.requests_by_day = {
        booking_day: [Request(booking_day, date(2025, 3, 10), 2, 1)]
    }
    monkeypatch.setattr(
        env, "_generate_prices", lambda requests, action: [[100.0, 120.0] for _ in requests]
    )
    plotted = []
    monkeypatch.setattr(
        module, "log_eval_artifact_plots", lambda artifact: plotted.append(artifact)
    )
    vec = VecMonitor(DummyVecEnv([lambda: env]))
    model = ZeroPolicy(vec)
    simulation_days = (end - booking_day).days + 1
    callback = (
        module.PlotCallback()
        if split == "train"
        else module.EvalOccupancyCallback(
            vec,
            n_eval_episodes=1,
            eval_freq=1,
            best_model_save_path=None,
            log_path=None,
            verbose=0,
        )
    )
    callback.init_callback(model)
    callback.on_training_start({}, {})
    payloads = []
    try:
        if split == "train":
            vec.reset()
            for day_index in range(simulation_days):
                _, rewards, dones, infos = vec.step(np.zeros((1, *vec.action_space.shape)))
                assert rewards[0] == (220.0 if day_index == 0 else 0.0)
                payloads.append(deepcopy(infos[0]["eval_artifacts"]))
                model.num_timesteps += 1
                callback.update_locals({"infos": infos, "dones": dones})
                assert callback.on_step()
        else:
            model.num_timesteps = simulation_days
            assert callback.on_step()
            assert len(artifacts) == len(plotted) == 1
            artifact = next(iter(artifacts.values()))
            assert plotted[0] == artifact
            payloads = artifact["steps"]
        callback.on_training_end()

        expected = [220.0] + [0.0] * (simulation_days - 1)
        assert [step["step_reward"] for step in payloads] == expected
        snapshot_metrics = build_snapshot_metrics(payloads, payloads[0])
        np.testing.assert_array_equal(snapshot_metrics["booked_revenue"], expected)
        np.testing.assert_array_equal(snapshot_metrics["day"], np.arange(1, simulation_days + 1))
        if end.month == 3 and end.day == 11:
            np.testing.assert_array_equal(
                snapshot_metrics["booked_price_points"], [[11, 1, 100], [12, 1, 120]]
            )
        else:
            assert snapshot_metrics["booked_price_points"].size == 0
        rows = list(csv.DictReader(StringIO(texts[f"episodes/{split}/episode_00000_reward.csv"])))
        assert [int(row["day"]) for row in rows] == list(range(1, simulation_days + 1))
        assert [float(row["reward"]) for row in rows] == expected
        daily = [
            (step, value)
            for key, value, step in metrics
            if key == f"reward/by_request_day/mean_completed_episodes/{split}"
        ]
        assert daily == list(enumerate(expected, start=1))
        assert points(metrics, "episode_return", split) == [(simulation_days, 220.0)]
    finally:
        vec.close()


def test_eval_callback_logs_once_per_round_and_preserves_episode_artifacts(
    make_env,
    logged,
):
    metrics, texts, artifacts = logged
    env, pricing = make_env(price=100.0, collect_artifacts=True)
    vec = VecMonitor(DummyVecEnv([lambda: env]))
    model = ZeroPolicy(vec)

    callback = module.EvalOccupancyCallback(
        vec,
        n_eval_episodes=2,
        eval_freq=2,
        best_model_save_path=None,
        log_path=None,
        verbose=0,
    )
    callback.init_callback(model)
    callback.on_training_start({}, {})

    try:
        # Simuliamo il contatore del training.
        # EvalCallback esegue realmente evaluate_policy() e gli episodi.
        for call in range(1, 6):
            if call == 4:
                # Cambiamo il prezzo soltanto nel test per rilevare
                # eventuali dati del primo checkpoint rimasti in memoria.
                pricing["nightly"] = 200.0

            model.num_timesteps = call * 4
            assert callback.on_step()

            if call == 1:
                assert not metrics
                assert not artifacts
            elif call in (2, 3):
                assert_temporal_metrics(metrics, "val", [(8, 100.0)])
                assert len(artifacts) == 2
            else:
                assert_temporal_metrics(metrics, "val", [(8, 100.0), (16, 200.0)])
                assert len(artifacts) == 4

        # SB3 ha valutato gli stessi episodi del callback.
        assert callback.last_mean_reward == pytest.approx(600.0)

        callback.on_training_end()
        assert_temporal_metrics(metrics, "val", [(8, 100.0), (16, 200.0)])

        assert env.today == START
        assert not env.occupancy.any()
        assert len(texts) == 16  # Quattro CSV per ciascuno dei quattro episodi.

        for checkpoint, price in [(8, 100.0), (16, 200.0)]:
            for episode in range(2):
                path = f"eval_data/{checkpoint:08d}_timesteps/env_00/episode_{episode:05d}.json"
                artifact = artifacts[path]

                assert artifact["checkpoint_timesteps"] == checkpoint
                assert artifact["episode_id"] == episode
                assert artifact["env_index"] == 0

                steps = artifact["steps"]
                assert [step["request_day"] for step in steps] == [
                    START.isoformat(),
                    END.isoformat(),
                ]
                assert [step["day_index"] for step in steps] == [0, 1]
                assert [step["step_reward"] for step in steps] == [
                    price,
                    2 * price,
                ]

                terminal_request = steps[-1]["requests"][0]
                assert terminal_request["confirmed"]
                assert terminal_request["check_in"] == "2026-01-02"
                assert terminal_request["daily_prices"] == [price, price]
    finally:
        vec.close()


def test_train_callback_logs_raw_action_statistics_and_figures(make_env, logged, monkeypatch):
    metrics, _, _ = logged
    figures = {}
    monkeypatch.setattr(module.mlflow, "log_figure", lambda fig, path: figures.update({path: fig}))
    env, _ = make_env()
    vec = VecMonitor(DummyVecEnv([lambda: env]))
    model = PPO("MultiInputPolicy", vec, n_steps=2, batch_size=2, n_epochs=1, seed=1, device="cpu")
    model.set_logger(Logger(folder=None, output_formats=[]))
    try:
        model.learn(total_timesteps=4, callback=module.PlotCallback(figure_every_rollouts=1))
    finally:
        vec.close()

    for name in (
        "action_mean",
        "action_std",
        "action_clipped_low_fraction",
        "action_clipped_high_fraction",
    ):
        series = points(metrics, name, "train")
        assert [step for step, _ in series] == [2, 4], name
    for name in ("action_clipped_low_fraction", "action_clipped_high_fraction"):
        assert all(0.0 <= value <= 1.0 for _, value in points(metrics, name, "train"))
    assert {"learning/action_mean.png", "learning/action_clipping.png"} <= set(figures)
    # 3 categories x 2 seasons, 4 coefficient lines each
    assert len(figures["learning/action_mean.png"].axes) == 6
    assert all(len(panel.lines) >= 4 for panel in figures["learning/action_mean.png"].axes)


def test_eval_callback_logs_learning_diagnostics(make_env, logged, monkeypatch, tmp_path):
    metrics, _, _ = logged
    figures = {}
    monkeypatch.setattr(module.mlflow, "log_figure", lambda fig, path: figures.update({path: fig}))
    monkeypatch.setattr(module, "log_eval_artifact_plots", lambda artifact: None)
    training_env, _ = make_env(price=100.0)
    evaluation_env, _ = make_env(price=200.0, collect_artifacts=True)
    training = VecNormalize(
        VecMonitor(DummyVecEnv([lambda: training_env])),
        norm_obs=False,
        norm_reward=True,
        gamma=0.995,
    )
    evaluation = VecNormalize(
        VecMonitor(DummyVecEnv([lambda: evaluation_env])),
        training=False,
        norm_obs=False,
        norm_reward=False,
        gamma=0.995,
    )
    model = PPO(
        "MultiInputPolicy",
        training,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        gamma=0.995,
        seed=42,
        device="cpu",
    )
    model.set_logger(Logger(folder=None, output_formats=[]))
    callback = module.EvalOccupancyCallback(
        evaluation,
        eval_freq=2,
        n_eval_episodes=2,
        verbose=0,
        baseline_returns={"constant_bar": [300.0, 320.0]},
    )
    try:
        model.learn(total_timesteps=4, callback=callback)
    finally:
        training.close()
        evaluation.close()

    # Raw (unnormalized) EUR returns, one list per evaluation round
    assert callback._eval_returns == {2: [600.0, 600.0], 4: [600.0, 600.0]}
    assert points(metrics, "baseline_return/constant_bar", "val") == [(2, 310.0), (4, 310.0)]
    assert [step for step, _ in points(metrics, "value_check/explained_variance", "val")] == [2, 4]
    assert {
        "learning/eval_return_vs_baselines.png",
        "learning/policy_curve_evolution.png",
        "value_check/00000002_timesteps/env_00/episode_00000.png",
        "value_check/00000002_timesteps/env_00/episode_00001.png",
        "value_check/00000004_timesteps/env_00/episode_00001.png",
    } <= set(figures)
    returns_axis = figures["learning/eval_return_vs_baselines.png"].axes[0]
    assert returns_axis.get_legend().get_texts()[-1].get_text() == "constant_bar: 310"
    np.testing.assert_allclose(returns_axis.lines[0].get_ydata(), [600.0, 600.0])
    policy_axes = figures["learning/policy_curve_evolution.png"].axes
    assert len(policy_axes) == 7  # 3 x 2 panels + colorbar
    assert all(len(panel.lines) == 2 for panel in policy_axes[:6])
