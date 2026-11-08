import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from pricing_advertising.plots.run_data import log_run_data
from pricing_advertising.training import multi_seed_training as module

SEEDS = [11, 22, 33]


def test_parent_financial_plots_average_episodes_then_seeds_per_checkpoint(monkeypatch):
    from pricing_advertising.plots import aggregate_plots

    def episode(value):
        calendar = {"start": "2025-01-01", "end": "2025-01-01", "day": [1]}
        return {
            "reward": {**calendar, "bookings": [value], "refunds": [10], "net": [value - 10]},
            "booked_revenue": {**calendar, "simulation": [value], "warmup": [5]},
        }

    evaluations = [
        {"financial_metrics": {"10": [episode(2), episode(4)], "20": [episode(100)]}},
        {"financial_metrics": {"10": [episode(9)], "20": [episode(200)]}},
        {},
    ]
    figures = {}
    monkeypatch.setattr(mlflow, "log_figure", lambda figure, path: figures.update({path: figure}))
    aggregate_plots.log_mean_financial_figures(evaluations)
    assert len(figures) == 4
    for checkpoint, expected in [(10, 6), (20, 150)]:
        reward = figures[f"reward_plots/{checkpoint:08d}_timesteps/mean_over_seeds.png"].axes[0]
        revenue = figures[
            f"booked_revenue_plots/{checkpoint:08d}_timesteps/mean_over_seeds.png"
        ].axes[0]
        np.testing.assert_allclose(reward.lines[0].get_ydata(), [expected - 10])
        np.testing.assert_allclose(revenue.lines[0].get_ydata(), [expected + 5])
        assert "Mean over 2 seeds" in reward.get_title()


def test_main_derives_evaluation_counts_from_seeds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    import json

    from pricing_advertising.__main__ import DEFAULT_CONFIG, main
    from pricing_advertising.pricing_model import bar_table

    calls = []

    def fake_simulation(**kwargs: object) -> float:
        calls.append(kwargs)
        return 0.0

    values = json.loads(DEFAULT_CONFIG.read_text())
    values["evaluation"]["eval_seeds"] = [100, 101, 102]
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(values))
    monkeypatch.setattr(module, "run_multi_seed_simulation", fake_simulation)
    monkeypatch.setattr(bar_table, "set_bar_table_from_csv", lambda path: None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow.get_tracking_uri())

    main(["--config", str(config_path)])

    assert calls[0]["eval_seeds"] == [100, 101, 102]
    assert "eval_episodes" not in calls[0]
    assert "n_eval_envs = 3" in capsys.readouterr().out


def snapshot_metrics(seed):
    size = 4
    return {
        "day": [10, 11, 12, 13],
        "requests": [2] * size,
        "offers": [2] * size,
        "confirmation_rate": [seed / 100] * size,
        "price_acceptance_rate": [0.5, None, 0.5, 0.5],
        "occupancy_pct": [seed] * size,
        "total_occupancy_pct": [seed * 2] * size,
        "bar_prices": [[100.0] * size, [150.0] * size],
    }


@pytest.mark.parametrize("legacy", [False, True])
def test_parent_cancellation_stacks_average_episodes_then_seeds(monkeypatch, legacy):
    from pricing_advertising.plots import aggregate_plots

    def episode(total, voided):
        values = snapshot_metrics(1)
        values["total_reservations"] = [[total] * 4] * 3
        values["voided_reservations"] = [[voided] * 4] * 3
        return values

    episodes = [episode(8, 4)] if not legacy else [snapshot_metrics(1)]
    evaluations = [
        {"snapshot_metrics": {"10": {"2025-01-09": [episode(2, 1), episode(4, 1)]}}},
        {"snapshot_metrics": {"10": {"2025-01-09": episodes}}},
    ]
    figures = {}
    monkeypatch.setattr(
        aggregate_plots.mlflow, "log_figure", lambda fig, path: figures.update({path: fig})
    )
    aggregate_plots.log_mean_snapshot_figures(evaluations)
    path = "cancellation_plots/2025-01-09/00000010_timesteps/mean_over_seeds.png"
    if legacy:
        assert path not in figures
        return
    figure = figures[path]
    assert len(figure.axes) == 3
    for panel in figure.axes:
        np.testing.assert_array_equal(panel.lines[0].get_ydata(), [5.5] * 4)
        assert panel.collections[0].get_paths()[0].vertices[:, 1].max() == 3.0
        assert panel.collections[1].get_paths()[0].vertices[:, 1].max() == 5.5


@pytest.fixture
def fake_runs(monkeypatch):
    def fake_run_simulation(*, seed, eval_seeds, run_name, **kwargs):
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_param("eval_seeds", str(eval_seeds))
            mlflow.log_metric("score", float(seed), step=1)
            # A step logged twice counts once, with its last value
            mlflow.log_metric("score", -1.0, step=2)
            mlflow.log_metric("score", float(seed) * 2, step=2)
            if seed != SEEDS[0]:
                mlflow.log_metric("only_in_some_seeds", 1.0, step=0)
            log_run_data(
                "evaluation",
                {
                    "eval_returns": {"10": [float(seed)], "20": [seed * 2.0]},
                    "baseline_returns": {"constant_bar": [1.0]},
                    "policy_curves": {"10": np.full((3, 2, 101), 0.5).tolist()},
                    "snapshot_metrics": {"10": {"2025-01-09": [snapshot_metrics(seed)]}},
                    "financial_metrics": {
                        "10": [
                            {
                                "reward": {
                                    "start": "2025-01-01",
                                    "end": "2025-01-01",
                                    "day": [1],
                                    "bookings": [float(seed)],
                                    "refunds": [0],
                                    "net": [float(seed)],
                                },
                                "booked_revenue": {
                                    "start": "2025-01-01",
                                    "end": "2025-01-01",
                                    "day": [1],
                                    "simulation": [float(seed)],
                                    "warmup": [0],
                                },
                            }
                        ]
                    },
                },
            )
        return float(seed)

    monkeypatch.setattr(module, "run_simulation", fake_run_simulation)


@pytest.mark.parametrize("options", [{}, {"use_subprocesses": True}, {"use_subprocesses": False}])
def test_multi_seed_forwards_subprocess_option(fake_runs, monkeypatch, options):
    calls = []
    fake_run = module.run_simulation

    def capture_run(**kwargs):
        calls.append(kwargs)
        return fake_run(**kwargs)

    monkeypatch.setattr(module, "run_simulation", capture_run)
    module.run_multi_seed_simulation(
        run_name="subprocess_option", seeds=SEEDS[:2], eval_seeds=[100, 101], **options
    )

    assert len(calls) == 2
    for call in calls:
        if "use_subprocesses" in options:
            assert call["use_subprocesses"] is options["use_subprocesses"]
        else:
            assert "use_subprocesses" not in call


def test_resolved_config_is_logged_only_on_parent(fake_runs):
    import json
    from pathlib import Path

    config = {"run": {"run_name": "configured"}, "ppo": {"n_envs": 1}}
    module.run_multi_seed_simulation(
        run_name="configured", seeds=SEEDS[:1], eval_seeds=[100], resolved_config=config
    )
    client = MlflowClient()
    experiment = client.get_experiment_by_name("PricingEnv_Simulation")
    runs = client.search_runs([experiment.experiment_id])
    parent = next(run for run in runs if run.info.run_name == "configured")
    child = next(run for run in runs if run.data.tags.get("mlflow.parentRunId"))
    artifact = client.download_artifacts(parent.info.run_id, "resolved_config.json")
    assert json.loads(Path(artifact).read_text()) == config
    assert "resolved_config" not in child.data.params
    assert "resolved_config.json" not in {
        item.path for item in client.list_artifacts(child.info.run_id)
    }


def test_parent_run_averages_child_metrics_and_logs_figures(fake_runs):
    reward = module.run_multi_seed_simulation(
        run_name="parent", seeds=SEEDS, eval_seeds=[100, 101], total_timesteps=5
    )

    assert reward == pytest.approx(np.mean(SEEDS))
    client = MlflowClient()
    experiment = client.get_experiment_by_name("PricingEnv_Simulation")
    runs = client.search_runs([experiment.experiment_id])
    parent = next(run for run in runs if run.info.run_name == "parent")
    children = [run for run in runs if run.data.tags.get("mlflow.parentRunId")]
    assert sorted(run.info.run_name for run in children) == [f"parent_seed_{s}" for s in SEEDS]
    assert all(run.data.tags["mlflow.parentRunId"] == parent.info.run_id for run in children)

    assert parent.data.params["seeds"] == str(SEEDS)
    assert parent.data.params["n_eval_envs"] == parent.data.params["eval_episodes"] == "2"
    assert parent.data.params["eval_seeds"] == "[100, 101]"
    assert all(child.data.params["eval_seeds"] == "[100, 101]" for child in children)
    assert parent.data.params["total_timesteps"] == "5"
    history = {m.step: m.value for m in client.get_metric_history(parent.info.run_id, "score")}
    assert history == {1: pytest.approx(np.mean(SEEDS)), 2: pytest.approx(2 * np.mean(SEEDS))}
    (partial,) = client.get_metric_history(parent.info.run_id, "only_in_some_seeds")
    assert partial.value == 1.0
    assert parent.data.metrics["final_eval_mean_reward_std_across_seeds"] == pytest.approx(
        np.std(SEEDS)
    )

    artifacts = {item.path for item in client.list_artifacts(parent.info.run_id)}
    assert {"learning", "occupancy_plots", "bar_price_plots"} <= artifacts
    learning = {item.path for item in client.list_artifacts(parent.info.run_id, "learning")}
    assert {
        "learning/eval_return_vs_baselines.png",
        "learning/policy_curve_evolution.png",
    } <= learning
    folder = "occupancy_plots/2025-01-09/00000010_timesteps"
    (figure,) = client.list_artifacts(parent.info.run_id, folder)
    assert figure.path == f"{folder}/mean_over_seeds.png"
    for kind in ("reward", "booked_revenue"):
        folder = f"{kind}_plots/00000010_timesteps"
        (figure,) = client.list_artifacts(parent.info.run_id, folder)
        assert figure.path == f"{folder}/mean_over_seeds.png"


@pytest.mark.parametrize("seeds", [[], [1, 1]])
def test_invalid_seeds_are_rejected(fake_runs, seeds):
    with pytest.raises(ValueError):
        module.run_multi_seed_simulation(run_name="parent", seeds=seeds)


@pytest.mark.parametrize("eval_seeds", [[], [100, 100]])
def test_invalid_eval_seeds_are_rejected_before_starting_runs(
    monkeypatch: pytest.MonkeyPatch, eval_seeds: list[int]
) -> None:
    def unexpected_run(**_kwargs: object) -> None:
        pytest.fail("Invalid evaluation seeds must be rejected before starting a run.")

    monkeypatch.setattr(mlflow, "start_run", unexpected_run)
    with pytest.raises(ValueError, match="evaluation seed|Evaluation seeds"):
        module.run_multi_seed_simulation(run_name="parent", seeds=SEEDS, eval_seeds=eval_seeds)
