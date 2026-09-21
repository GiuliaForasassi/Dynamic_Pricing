from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.plots.eval_artifact_plots import log_eval_artifact_plots
from pricing_advertising.pricing_model import bar_table


@pytest.mark.parametrize(
    "start,end,expected_days",
    [
        (date(2025, 1, 1), date(2025, 12, 31), 365),
        (date(2024, 1, 1), date(2024, 12, 31), 366),
        (date(2025, 12, 31), date(2025, 12, 31), 1),
    ],
)
def test_environment_includes_last_day(monkeypatch, start, end, expected_days):
    csv_path = Path(__file__).resolve().parents[1] / "data" / "bar_table.csv"
    monkeypatch.setattr(
        bar_table,
        "BAR_TABLE",
        bar_table.load_bar_table_wide_csv(csv_path),
    )

    env = PricingEnv(
        start_simulation=start,
        end_simulation=end,
        hotel=Hotel(
            num_category1_rooms=100,
            num_category2_rooms=100,
            num_category3_rooms=100,
        ),
        avg_num_requests=5,
        seed=42,
        collect_eval_artifacts=True,
        artifact_snapshot_dates=[end],
    )

    try:
        env.reset(seed=42)

        assert env.total_days == expected_days
        assert (env.extended_end_simulation - start).days == env.extended_simulation_days
        assert len(env.seasonality_array) == env.extended_simulation_days
        assert env.occupancy.shape[1] == env.extended_simulation_days

        # Con avg_num_requests=5 il generatore attuale produce
        # almeno una richiesta al giorno.
        last_requests = env.request_generator.fetch_requests_for_date(end)
        assert len(last_requests) > 0
        assert all(request.t_request == end for request in last_requests)

        after_end = end + timedelta(days=1)
        assert len(env.request_generator.fetch_requests_for_date(after_end)) == 0

        action = np.zeros(env.action_space.shape, dtype=np.float32)

        for index in range(expected_days):
            observation, _, terminated, truncated, info = env.step(action)
            payload = info["eval_artifacts"]

            assert payload["request_day"] == (start + timedelta(days=index)).isoformat()
            assert payload["day_index"] == index
            assert not truncated
            assert bool(terminated) == (index == expected_days - 1)
            assert env.observation_space.contains(observation)

        # L'ultimo step deve essere proprio end_simulation.
        assert payload["request_day"] == end.isoformat()
        assert len(payload["requests"]) > 0
        assert env.today == after_end

        # La fotografia dell'ultimo giorno contiene una sola notte.
        assert payload["snapshot_start_day"] == end.isoformat()
        assert np.asarray(payload["occupancy_by_category"]).shape == (3, 1)
        assert np.asarray(payload["bar_prices_by_category"]).shape == (3, 1)
    finally:
        env.close()


@pytest.mark.parametrize(
    "num_steps,should_pass",
    [(2, True), (1, False)],
)
def test_artifact_validation_requires_last_day(num_steps, should_pass):
    start = date(2025, 12, 30)
    end = date(2025, 12, 31)

    artifact = {
        "schema_version": 3,
        "decision_end_inclusive": True,
        "checkpoint_timesteps": 800,
        "env_index": 0,
        "episode_id": 0,
        # Nessuna fotografia: verifica la durata senza generare PNG.
        "snapshot_dates": [],
        "steps": [
            {
                "start_simulation": start.isoformat(),
                "end_simulation": end.isoformat(),
                "request_day": (start + timedelta(days=index)).isoformat(),
                "day_index": index,
            }
            for index in range(num_steps)
        ],
    }

    if should_pass:
        log_eval_artifact_plots(artifact)
    else:
        with pytest.raises(ValueError, match="episodio incompleto"):
            log_eval_artifact_plots(artifact)


def test_step_metrics_survive_vectorized_reset(monkeypatch):
    from stable_baselines3.common.vec_env import DummyVecEnv

    from pricing_advertising.requests.request import Request

    csv_path = Path(__file__).resolve().parents[1] / "data" / "bar_table.csv"
    monkeypatch.setattr(
        bar_table,
        "BAR_TABLE",
        bar_table.load_bar_table_wide_csv(csv_path),
    )

    start = date(2025, 12, 30)
    end = date(2025, 12, 31)

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
        collect_eval_artifacts=False,
    )

    # Il 30 dicembre prenotiamo la notte del 31.
    # Il 31 dicembre prenotiamo due notti di gennaio.
    env.request_generator.requests_by_day = {
        start: [Request(start, end, 1, 1)],
        end: [Request(end, date(2026, 1, 2), 2, 1)],
    }

    # Prezzi e accettazioni deterministici: isoliamo il test
    # dalla casualità della policy e del modello di accettazione.
    monkeypatch.setattr(
        env,
        "_generate_prices",
        lambda requests, action: [[100.0] * int(request.days_stay) for request in requests],
    )
    monkeypatch.setattr(
        env,
        "_evaluate_prices",
        lambda prices, requests: (
            np.ones(len(requests), dtype=bool),
            np.asarray([sum(values) for values in prices], dtype=float),
        ),
    )

    vec = DummyVecEnv([lambda: env])

    try:
        vec.reset()
        action = np.zeros((1, *env.action_space.shape), dtype=np.float32)

        _, _, dones, infos = vec.step(action)
        assert not dones[0]
        assert infos[0]["step_metrics"]["day"] == start
        assert infos[0]["step_metrics"]["mean_check_in_offset"] == 1.0

        _, rewards, dones, infos = vec.step(action)
        assert dones[0]

        metrics = infos[0]["step_metrics"]

        # L'ambiente è già resettato...
        assert env.today == start
        assert not env.occupancy.any()

        # ...ma il messaggio conserva i dati del 31 dicembre.
        assert metrics["day"] == end
        assert metrics["day_index"] == 1
        assert metrics["occupancy_pct"] == pytest.approx(100.0 / 6)
        assert metrics["num_requests"] == 1
        assert metrics["request_stays"] == [(3, 2)]
        assert metrics["mean_check_in_offset"] == 3.0
        assert metrics["num_offers"] == 1
        assert metrics["num_price_accepted"] == 1
        assert metrics["num_confirmed"] == 1

        # La prenotazione di gennaio contribuisce al reward,
        # pur essendo fuori dall'intervallo dei grafici.
        assert metrics["reward"] == pytest.approx(200.0)
        assert rewards[0] == pytest.approx(200.0)

        np.testing.assert_array_equal(
            metrics["confirmed_nightly_prices"],
            [100.0, 100.0],
        )
        assert metrics["confirmed_requests"][0].t_reservation == date(2026, 1, 2)
        assert "eval_artifacts" not in infos[0]
    finally:
        vec.close()


@pytest.mark.parametrize("avg_num_requests", [5, None])
@pytest.mark.parametrize(
    "second_seed,same_requests",
    [(42, True), (43, False)],
)
def test_request_scenarios_follow_constructor_seed(
    monkeypatch,
    avg_num_requests,
    second_seed,
    same_requests,
):
    csv_path = Path(__file__).resolve().parents[1] / "data" / "bar_table.csv"
    monkeypatch.setattr(
        bar_table,
        "BAR_TABLE",
        bar_table.load_bar_table_wide_csv(csv_path),
    )

    def make_env(seed):
        return PricingEnv(
            start_simulation=date(2025, 1, 1),
            end_simulation=date(2025, 1, 3),
            hotel=Hotel(
                num_category1_rooms=2,
                num_category2_rooms=2,
                num_category3_rooms=2,
            ),
            avg_num_requests=avg_num_requests,
            seed=seed,
        )

    def request_signature(env):
        return tuple(
            (
                day,
                tuple(
                    (
                        request.t_request,
                        request.t_reservation,
                        int(request.days_stay),
                        int(request.num_guests),
                    )
                    for request in requests
                ),
            )
            for day, requests in sorted(env.request_generator.requests_by_day.items())
        )

    first = make_env(42)
    second = make_env(second_seed)

    try:
        first_signature = request_signature(first)
        second_signature = request_signature(second)

        assert (first_signature == second_signature) == same_requests

        if same_requests:
            assert first.avg_num_requests == second.avg_num_requests

        if avg_num_requests is not None:
            assert first.avg_num_requests == avg_num_requests
            assert second.avg_num_requests == avg_num_requests

        # Il reset ordinario non sostituisce il calendario pregenerato,
        # neppure quando cambia il seed delle altre dinamiche.
        first.reset(seed=999)
        assert request_signature(first) == first_signature

        first.reset()
        assert request_signature(first) == first_signature
    finally:
        first.close()
        second.close()
