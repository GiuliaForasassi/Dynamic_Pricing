"""Figures of the parent run of a multi-seed experiment: averages of the per-seed runs."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from matplotlib.figure import Figure

from pricing_advertising.plots.alternatives_plots import log_alternative_figures
from pricing_advertising.plots.eval_artifact_plots import RATE_COUNT_KEYS
from pricing_advertising.plots.learning_plots import (
    ActionStatistics,
    action_clipping_figure,
    action_mean_figure,
    eval_return_figure,
    policy_evolution_figure,
)
from pricing_advertising.pricing_model.bar_table import max_price

if TYPE_CHECKING:
    from collections.abc import Sequence


def _nanmean(values: Sequence[Any], axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.asarray(values, dtype=float), axis=axis)


def _nanstd(values: Sequence[Any], axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(np.asarray(values, dtype=float), axis=axis)


def _log_figure(fig: Figure, artifact_file: str) -> None:
    try:
        mlflow.log_figure(fig, artifact_file)
    finally:
        plt.close(fig)


# ----------- LEARNING FIGURES -----------
def _mean_action_statistics(histories: Sequence[ActionStatistics]) -> ActionStatistics:
    length = min(len(history.timesteps) for history in histories)
    return ActionStatistics(
        timesteps=histories[0].timesteps[:length],
        **{
            name: [
                _nanmean([getattr(history, name)[index] for history in histories])
                for index in range(length)
            ]
            for name in ("mean", "std", "clipped_low", "clipped_high")
        },
    )


def log_mean_learning_figures(
    train_actions: Sequence[dict[str, list]], evaluations: Sequence[dict[str, Any]]
) -> None:
    if train_actions:
        statistics = _mean_action_statistics([ActionStatistics.from_dict(d) for d in train_actions])
        _log_figure(action_mean_figure(statistics), "learning/action_mean.png")
        _log_figure(action_clipping_figure(statistics), "learning/action_clipping.png")

    if not evaluations:
        return

    # One point per seed: the mean of that seed's evaluation episodes
    returns: dict[int, list[float]] = {}
    for evaluation in evaluations:
        for step, values in evaluation["eval_returns"].items():
            returns.setdefault(int(step), []).append(float(np.mean(values)))
    baselines: dict[str, list[float]] = {}
    for evaluation in evaluations:
        for name, values in evaluation["baseline_returns"].items():
            baselines.setdefault(name, []).extend(values)
    if returns:
        _log_figure(
            eval_return_figure(
                returns,
                baselines,
                title=(
                    f"Evaluation return against baselines: mean +/- std over {len(evaluations)} "
                    "seeds (one point per seed)"
                ),
            ),
            "learning/eval_return_vs_baselines.png",
        )

    curves: dict[int, list[np.ndarray]] = {}
    for evaluation in evaluations:
        for step, curve in evaluation["policy_curves"].items():
            curves.setdefault(int(step), []).append(np.asarray(curve, dtype=float))
    if curves:
        mean_curves = {step: _nanmean(values) for step, values in curves.items()}
        _log_figure(policy_evolution_figure(mean_curves), "learning/policy_curve_evolution.png")


# ----------- ALTERNATIVES FIGURES -----------
def log_mean_alternative_figures(split: str, runs: Sequence[dict[str, Any]]) -> None:
    if not runs:
        return
    mean = {
        name: _nanmean([run[name] for run in runs])
        for name in ("totals", "by_checkin", "occupancy_pct", "requests")
    }
    log_alternative_figures(
        split, **mean, num_episodes=sum(int(run["num_episodes"]) for run in runs)
    )


# ----------- EVALUATION SNAPSHOT FIGURES -----------
def _band(ax: Any, x: np.ndarray, values: np.ndarray, **kwargs: Any) -> None:  # noqa: ANN401
    mean = _nanmean(values)
    spread = _nanstd(values)
    (line,) = ax.plot(x, mean, **kwargs)
    ax.fill_between(x, mean - spread, mean + spread, color=line.get_color(), alpha=0.15)


def _snapshot_figure(
    key: str,
    folder: str,
    ylabel: str,
    ylim: tuple[float, float | None] | None,
    *,
    metrics: dict[str, np.ndarray],
    cutoff: str,
    checkpoint: int,
    n_seeds: int,
) -> None:
    x = metrics["day"][0]
    fig = Figure(figsize=(12, 5), layout="constrained")
    ax = fig.subplots()
    if key == "bar_prices":
        for category, prices in enumerate(np.moveaxis(metrics[key], 1, 0), start=1):
            _band(ax, x, prices, label=f"Category {category}")
        ax.legend()
    elif key in RATE_COUNT_KEYS:
        mean, spread = _nanmean(metrics[key]), _nanstd(metrics[key])
        counts = _nanmean(metrics[RATE_COUNT_KEYS[key]])
        observed = np.isfinite(mean)
        sizes = 8 + 52 * counts[observed] / max(counts[observed].max(initial=1), 1)
        ax.errorbar(x[observed], mean[observed], yerr=spread[observed], fmt="none", alpha=0.25)
        ax.scatter(x[observed], mean[observed], s=sizes, alpha=0.7)
        if not observed.any():
            ax.text(0.5, 0.5, "No observations", transform=ax.transAxes, ha="center")
    else:
        _band(ax, x, metrics[key], marker=".", markersize=2, label="Added by today's bookings")
        _band(ax, x, metrics["total_occupancy_pct"], label="Total hotel occupancy")
        ax.legend(loc="upper left")
    ax.set(ylabel=ylabel, xlim=(float(x[0]) - 0.5, float(x[-1]) + 0.5))
    ax.set_xlabel(
        "Stay day (1 = simulation start)"
        + (
            f" | marker size = mean number of {RATE_COUNT_KEYS[key]}"
            if key in RATE_COUNT_KEYS
            else ""
        )
    )
    if key == "bar_prices":
        ylim = (0, max_price() or None)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"{ylabel}\nMean over {n_seeds} seeds (band = +/- 1 std) | Requests received only on "
        f"{cutoff} | Training timesteps: {checkpoint}"
    )
    _log_figure(fig, f"{folder}/{cutoff}/{checkpoint:08d}_timesteps/mean_over_seeds.png")


_SNAPSHOT_FIGURES = (
    (
        "confirmation_rate",
        "confirmation_rate_plots",
        "Confirmation rate (originally requested nights)\nConfirmed requests / all requests",
        (-0.05, 1.05),
    ),
    (
        "price_acceptance_rate",
        "price_acceptance_plots",
        "Price acceptance rate (offered nights)\nPrice-accepted offers / all offers",
        (-0.05, 1.05),
    ),
    (
        "occupancy_pct",
        "occupancy_plots",
        "Occupancy added by today's bookings (%) (confirmed offered nights)",
        (0, 100),
    ),
    ("bar_prices", "bar_price_plots", "Current BAR nightly price", (0, None)),
)


def log_mean_snapshot_figures(evaluations: Sequence[dict[str, Any]]) -> None:
    checkpoints = sorted({int(step) for ev in evaluations for step in ev["snapshot_metrics"]})
    for checkpoint in checkpoints:
        by_seed = [ev["snapshot_metrics"].get(str(checkpoint), {}) for ev in evaluations]
        for cutoff in sorted({date for snapshots in by_seed for date in snapshots}):
            # One entry per seed: the mean of that seed's evaluation episodes
            seeds = [
                {
                    key: _nanmean([episode[key] for episode in snapshots[cutoff]])
                    for key in snapshots[cutoff][0]
                }
                for snapshots in by_seed
                if snapshots.get(cutoff)
            ]
            metrics = {key: np.stack([seed[key] for seed in seeds]) for key in seeds[0]}
            for key, folder, ylabel, ylim in _SNAPSHOT_FIGURES:
                _snapshot_figure(
                    key,
                    folder,
                    ylabel,
                    ylim,
                    metrics=metrics,
                    cutoff=cutoff,
                    checkpoint=checkpoint,
                    n_seeds=len(seeds),
                )
