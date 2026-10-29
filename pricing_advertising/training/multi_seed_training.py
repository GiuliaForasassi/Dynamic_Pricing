"""One main execution = several seeded PPO runs (MLflow child runs) averaged in the parent run."""

import time
from collections.abc import Sequence
from typing import Any

import mlflow
import numpy as np
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

from pricing_advertising.plots.aggregate_plots import (
    log_mean_alternative_figures,
    log_mean_learning_figures,
    log_mean_snapshot_figures,
)
from pricing_advertising.plots.run_data import load_run_data
from pricing_advertising.training.ppo_training import run_simulation
from pricing_advertising.utils.constants import EVAL_SEEDS, SEEDS

# MLflow accepts at most 1000 metrics per log_batch call
_BATCH_SIZE = 1000


def average_child_metrics(
    client: MlflowClient, parent_run_id: str, child_run_ids: Sequence[str]
) -> None:
    """Log on the parent run, for every metric key and step, the mean over the child runs."""
    values: dict[str, dict[int, list[float]]] = {}
    for run_id in child_run_ids:
        for key in client.get_run(run_id).data.metrics:
            # A step logged more than once in a run counts once, with its last value
            last_by_step = {
                metric.step: metric.value
                for metric in sorted(
                    client.get_metric_history(run_id, key), key=lambda metric: metric.timestamp
                )
            }
            for step, value in last_by_step.items():
                if np.isfinite(value):
                    values.setdefault(key, {}).setdefault(step, []).append(value)

    now = int(time.time() * 1000)
    metrics = [
        Metric(key, float(np.mean(step_values)), now, step)
        for key, steps in values.items()
        for step, step_values in sorted(steps.items())
    ]
    for start in range(0, len(metrics), _BATCH_SIZE):
        client.log_batch(parent_run_id, metrics=metrics[start : start + _BATCH_SIZE])


def _load_all(run_ids: Sequence[str], name: str) -> list[dict[str, Any]]:
    loaded = (load_run_data(run_id, name) for run_id in run_ids)
    return [data for data in loaded if data is not None]


def run_multi_seed_simulation(
    *,
    run_name: str,
    seeds: Sequence[int] = SEEDS,
    eval_seeds: Sequence[int] = EVAL_SEEDS,
    **simulation_kwargs: Any,  # noqa: ANN401
) -> float:
    """Run `run_simulation` once per seed, each in its own child run.

    All child runs are evaluated on the same `eval_seeds` environments.
    The parent run holds the mean over seeds of every child metric and of the evaluation, learning
    and alternatives figures. Returns the mean over seeds of the final evaluation reward.
    """
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct.")
    if not eval_seeds:
        raise ValueError("At least one evaluation seed is required.")
    if len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError("Evaluation seeds must be distinct.")

    mlflow.set_experiment("PricingEnv_Simulation")
    client = MlflowClient()
    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.log_params(
            {
                **simulation_kwargs,
                "seeds": str(list(seeds)),
                "n_seeds": len(seeds),
                "eval_seeds": str(list(eval_seeds)),
                "n_eval_envs": len(eval_seeds),
                "eval_episodes": len(eval_seeds),
            }
        )
        rewards = [
            run_simulation(
                seed=seed,
                eval_seeds=list(eval_seeds),
                run_name=f"{run_name}_seed_{seed}",
                **simulation_kwargs,
            )
            for seed in seeds
        ]

        children = client.search_runs(
            [parent.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
            max_results=len(seeds),
            order_by=["attributes.start_time ASC"],
        )
        child_run_ids = [child.info.run_id for child in children]

        average_child_metrics(client, parent.info.run_id, child_run_ids)
        mlflow.log_metric("final_eval_mean_reward_std_across_seeds", float(np.std(rewards)))

        evaluations = _load_all(child_run_ids, "evaluation")
        log_mean_learning_figures(_load_all(child_run_ids, "train_actions"), evaluations)
        for split in ("train", "val"):
            log_mean_alternative_figures(split, _load_all(child_run_ids, f"alternatives_{split}"))
        log_mean_snapshot_figures(evaluations)

        return float(np.mean(rewards))
