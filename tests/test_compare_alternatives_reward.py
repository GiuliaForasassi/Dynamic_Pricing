import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from pricing_advertising.diagnostics import compare_alternatives_reward as module


def log_run(client, name, rewards_by_episode, requests=(5, 6, 7)):
    experiment = client.get_experiment_by_name(module.EXPERIMENT_NAME)
    experiment_id = (
        experiment.experiment_id if experiment else client.create_experiment(module.EXPERIMENT_NAME)
    )
    run = client.create_run(experiment_id, run_name=name)
    run_id = run.info.run_id
    for episode_id, rewards in rewards_by_episode.items():
        text = "day,reward\n" + "\n".join(f"{day},{r}" for day, r in enumerate(rewards, 1))
        client.log_text(run_id, text, f"episodes/val/episode_{episode_id:05d}_reward.csv")
        counts = "day,requests_arriving,requests_covering_night\n" + "\n".join(
            f"{day},{n},0" for day, n in enumerate(requests, 1)
        )
        client.log_text(run_id, counts, f"episodes/val/episode_{episode_id:05d}_request_counts.csv")
    client.set_terminated(run_id)
    return run_id


@pytest.fixture
def store(tmp_path):
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


@pytest.fixture
def client(store):
    return MlflowClient(store)


def test_compare_uses_the_final_episodes_and_checks_requests(client):
    with_id = log_run(client, "with", {0: [1, 1, 1], 1: [10, 20, 30]})
    without_id = log_run(client, "without", {0: [9, 9, 9], 1: [10, 10, 30]})

    result = module.compare_runs(client, with_id, without_id, last_episodes=1)

    assert result["with"].tolist() == [10, 20, 30]
    assert result["without"].tolist() == [10, 10, 30]
    assert result["same_requests"]


def test_different_requests_are_reported(client):
    with_id = log_run(client, "with", {0: [1, 2, 3]}, requests=(1, 1, 1))
    without_id = log_run(client, "without", {0: [1, 2, 3]}, requests=(2, 2, 2))

    result = module.compare_runs(client, with_id, without_id, last_episodes=1)

    assert not result["same_requests"]


def test_runs_with_different_episodes_are_rejected(client):
    with_id = log_run(client, "with", {0: [1, 2, 3]})
    without_id = log_run(client, "without", {1: [1, 2, 3]})

    with pytest.raises(ValueError, match="different validation episodes"):
        module.compare_runs(client, with_id, without_id, last_episodes=1)


def test_cli_resolves_run_names_and_saves_figure(client, store, tmp_path):
    log_run(client, "with", {0: [1, 5, 3]})
    log_run(client, "without", {0: [1, 2, 3]})
    output = tmp_path / "out" / "diff.png"

    module.main(
        [
            "--with-alternatives",
            "with",
            "--without-alternatives",
            "without",
            "--tracking-uri",
            store,
            "--output",
            str(output),
        ]
    )

    assert output.stat().st_size > 0


def test_unknown_run_is_rejected(client):
    log_run(client, "existing", {0: [1, 2, 3]})
    with pytest.raises(ValueError, match="Run not found"):
        module.resolve_run_id(client, module.EXPERIMENT_NAME, "missing")
    with pytest.raises(ValueError, match="Experiment not found"):
        module.resolve_run_id(client, "other_experiment", "existing")


def test_difference_figure_counts_changed_days():
    fig = module.plot_reward_difference(np.array([1.0, 5.0, 3.0]), np.array([1.0, 2.0, 3.0]))

    assert "days with a different reward: 1/3" in fig._suptitle.get_text()
    assert "WARNING" not in fig._suptitle.get_text()
    with pytest.raises(ValueError, match="different number of days"):
        module.plot_reward_difference(np.zeros(2), np.zeros(3))
