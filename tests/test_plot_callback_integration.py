import csv
from copy import deepcopy
from datetime import date
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.plots import plot_callback as module
from pricing_advertising.pricing_model import bar_table
from pricing_advertising.requests.request import Request

START = date(2025, 12, 30)
END = date(2025, 12, 31)


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


@pytest.fixture
def make_env(monkeypatch):
    csv_path = Path(__file__).resolve().parents[1] / "data" / "bar_table.csv"
    monkeypatch.setattr(
        bar_table,
        "BAR_TABLE",
        bar_table.load_bar_table_wide_csv(csv_path),
    )

    def factory(price=100.0, collect_artifacts=False):
        env = PricingEnv(
            start_simulation=START,
            end_simulation=END,
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
            START: [Request(START, END, 1, 1)],
            END: [Request(END, date(2026, 1, 2), 2, 1)],
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
        assert not artifacts
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
