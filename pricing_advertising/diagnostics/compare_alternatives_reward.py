"""Confronto giornaliero del reward tra due run: con e senza alternative.

Usa gli ultimi episodi di validation salvati da `RewardTracker` (policy finale). Se le due run
usano lo stesso `eval_seed` le richieste sono identiche, quindi la differenza giorno per giorno
è appaiata: mostra in quali giorni le alternative cambiano davvero il reward.
Il confronto delle richieste (`RequestTracker`) segnala se questa ipotesi non è valida.

Uso (dalla root del progetto):
    uv run python -m pricing_advertising.diagnostics.compare_alternatives_reward \
        --with-alternatives PPO_1M_with_alternatives \
        --without-alternatives PPO_1M_no_alternatives
"""

from __future__ import annotations

import argparse
import csv
import re
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from loguru import logger
from matplotlib.figure import Figure
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "PricingEnv_Simulation"
_EPISODE_FILE = re.compile(r"episode_(\d+)_(reward|request_counts)\.csv$")


# ---------------- RUN LOOKUP ----------------
def resolve_run_id(client: MlflowClient, experiment_name: str, run: str) -> str:
    """Accetta un run id oppure un run name (in tal caso prende il più recente)."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")
    found = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"attributes.run_id = '{run}'",
        max_results=1,
    ) or client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not found:
        raise ValueError(f"Run not found in {experiment_name}: {run}")
    return str(found[0].info.run_id)


# ---------------- EPISODE DATA ----------------
def _read_column(text: str, column: str) -> np.ndarray:
    rows = list(csv.DictReader(StringIO(text)))
    return np.asarray([float(row[column]) for row in rows], dtype=float)


def load_last_episodes(
    client: MlflowClient, run_id: str, kind: str, column: str, last_episodes: int
) -> dict[int, np.ndarray]:
    """Per ogni episodio di validation più recente, la colonna `column` del CSV `kind`."""
    ids = []
    for artifact in client.list_artifacts(run_id, "episodes/val"):
        match = _EPISODE_FILE.search(artifact.path)
        if match and match.group(2) == kind:
            ids.append(int(match.group(1)))
    if not ids:
        raise ValueError(f"No validation episodes ({kind}) in run {run_id}.")
    out = {}
    for episode_id in sorted(ids)[-last_episodes:]:
        path = f"episodes/val/episode_{episode_id:05d}_{kind}.csv"
        with TemporaryDirectory() as directory:
            local_path = client.download_artifacts(run_id, path, directory)
            out[episode_id] = _read_column(Path(local_path).read_text(), column)
    return out


def compare_runs(
    client: MlflowClient,
    with_run_id: str,
    without_run_id: str,
    last_episodes: int,
) -> dict[str, np.ndarray | bool]:
    rewards = [
        load_last_episodes(client, run_id, "reward", "reward", last_episodes)
        for run_id in (with_run_id, without_run_id)
    ]
    if rewards[0].keys() != rewards[1].keys():
        raise ValueError(
            "The runs have different validation episodes: "
            f"{sorted(rewards[0])} vs {sorted(rewards[1])}."
        )
    requests = [
        load_last_episodes(client, run_id, "request_counts", "requests_arriving", last_episodes)
        for run_id in (with_run_id, without_run_id)
    ]
    same_requests = all(
        np.array_equal(requests[0][episode_id], requests[1][episode_id])
        for episode_id in requests[0]
    )
    return {
        "with": np.mean(list(rewards[0].values()), axis=0),
        "without": np.mean(list(rewards[1].values()), axis=0),
        "same_requests": same_requests,
    }


# ---------------- FIGURE ----------------
def plot_reward_difference(
    with_rewards: np.ndarray,
    without_rewards: np.ndarray,
    *,
    with_label: str = "With alternatives",
    without_label: str = "Without alternatives",
    same_requests: bool = True,
) -> Figure:
    if with_rewards.shape != without_rewards.shape:
        raise ValueError("The two runs have a different number of days.")
    days = np.arange(1, len(with_rewards) + 1)
    difference = with_rewards - without_rewards

    fig = Figure(figsize=(14, 10), layout="constrained")
    reward_ax, difference_ax, cumulative_ax = fig.subplots(3, 1, sharex=True)
    reward_ax.plot(days, with_rewards, label=with_label, color="#c04cfd", linewidth=1)
    reward_ax.plot(days, without_rewards, label=without_label, color="#ffa500", linewidth=1)
    reward_ax.set_ylabel("Daily reward")
    reward_ax.legend()

    difference_ax.bar(
        days,
        difference,
        width=1.0,
        color=np.where(difference >= 0, "#2ca02c", "#d62728"),
    )
    difference_ax.axhline(0, color="black", linewidth=0.8)
    difference_ax.set_ylabel("Reward with - without")

    cumulative_ax.plot(days, np.cumsum(difference), color="black")
    cumulative_ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    cumulative_ax.set(xlabel="Request day", ylabel="Cumulative difference")

    for axis in (reward_ax, difference_ax, cumulative_ax):
        axis.grid(True, alpha=0.3)

    total_with = float(with_rewards.sum())
    total_without = float(without_rewards.sum())
    relative = (total_with - total_without) / total_without * 100 if total_without else float("nan")
    title = (
        f"Total reward: {total_with:,.0f} vs {total_without:,.0f} ({relative:+.2f}%), "
        f"days with a different reward: {int(np.count_nonzero(difference))}/{len(days)}"
    )
    if not same_requests:
        title += (
            "\nWARNING: the runs did not receive the same requests, the difference is not paired"
        )
    fig.suptitle(title)
    return fig


# ---------------- CLI ----------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--with-alternatives", required=True, help="Run name or run id")
    parser.add_argument("--without-alternatives", required=True, help="Run name or run id")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument("--tracking-uri", default=None, help="Default: MLFLOW_TRACKING_URI")
    parser.add_argument(
        "--last-episodes",
        type=int,
        default=1,
        help="Number of final validation episodes averaged for each run",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("tmp/alternatives_reward_difference.png")
    )
    args = parser.parse_args(argv)
    if args.last_episodes < 1:
        parser.error("--last-episodes must be positive")

    client = MlflowClient(tracking_uri=args.tracking_uri)
    with_id = resolve_run_id(client, args.experiment, args.with_alternatives)
    without_id = resolve_run_id(client, args.experiment, args.without_alternatives)

    result = compare_runs(client, with_id, without_id, args.last_episodes)
    same_requests = bool(result["same_requests"])
    if not same_requests:
        logger.warning("The runs did not receive the same validation requests.")
    fig = plot_reward_difference(
        np.asarray(result["with"]),
        np.asarray(result["without"]),
        with_label=args.with_alternatives,
        without_label=args.without_alternatives,
        same_requests=same_requests,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120)
    logger.info(f"Saved {args.output}")


if __name__ == "__main__":
    main()
