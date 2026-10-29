"""Figures and statistics that show whether the policy is learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from pricing_advertising.pricing_model.bernstein_model import BernsteinProbModel

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.figure import Figure

OCCUPANCY_GRID = np.linspace(0.0, 1.0, 101)
SEASON_LABELS = ("Low season", "High season")


# ----------- TRAINING ACTION STATISTICS -----------
@dataclass
class ActionStatistics:
    """Per-rollout statistics of the raw (unclipped) actions sampled by the policy."""

    timesteps: list[int] = field(default_factory=list)
    mean: list[np.ndarray] = field(default_factory=list)
    std: list[np.ndarray] = field(default_factory=list)
    clipped_low: list[np.ndarray] = field(default_factory=list)
    clipped_high: list[np.ndarray] = field(default_factory=list)

    # `actions` has shape (samples, category, season, coefficient); returns overall scalars to log
    def add(
        self,
        timestep: int,
        actions: np.ndarray,
        low: float | np.ndarray,
        high: float | np.ndarray,
    ) -> dict[str, float]:
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 4 or actions.shape[0] == 0:
            raise ValueError("Actions must have shape (samples, category, season, coefficient).")
        self.timesteps.append(int(timestep))
        self.mean.append(actions.mean(axis=0))
        self.std.append(actions.std(axis=0))
        self.clipped_low.append((actions <= low).mean(axis=0))
        self.clipped_high.append((actions >= high).mean(axis=0))
        return {
            "action_mean": float(self.mean[-1].mean()),
            "action_std": float(self.std[-1].mean()),
            "action_clipped_low_fraction": float(self.clipped_low[-1].mean()),
            "action_clipped_high_fraction": float(self.clipped_high[-1].mean()),
        }

    def to_dict(self) -> dict[str, list]:
        return {
            "timesteps": list(self.timesteps),
            **{
                name: [value.tolist() for value in getattr(self, name)]
                for name in ("mean", "std", "clipped_low", "clipped_high")
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, list]) -> ActionStatistics:
        return cls(
            timesteps=[int(step) for step in data["timesteps"]],
            **{
                name: [np.asarray(value, dtype=float) for value in data[name]]
                for name in ("mean", "std", "clipped_low", "clipped_high")
            },
        )


def _category_season_grid(
    title: str, ylabel: str, num_categories: int
) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(
        num_categories, 2, figsize=(13, 3.4 * num_categories), sharex=True, squeeze=False
    )
    for category_idx in range(num_categories):
        for season, label in enumerate(SEASON_LABELS):
            axes[category_idx, season].set_title(f"Category {category_idx + 1} | {label}")
            axes[category_idx, season].grid(True, alpha=0.25)
        axes[category_idx, 0].set_ylabel(ylabel)
    for panel in axes[-1]:
        panel.set_xlabel("Training timesteps")
    fig.suptitle(title)
    return fig, axes


def action_mean_figure(history: ActionStatistics) -> Figure:
    """Mean +/- std of each raw Bernstein coefficient during training."""
    timesteps = np.asarray(history.timesteps)
    mean = np.stack(history.mean)
    std = np.stack(history.std)
    fig, axes = _category_season_grid(
        "Sampled actions before clipping: mean +/- std per rollout",
        "Coefficient",
        mean.shape[1],
    )
    for category_idx in range(mean.shape[1]):
        for season in range(2):
            panel = axes[category_idx, season]
            for coefficient in range(mean.shape[3]):
                values = mean[:, category_idx, season, coefficient]
                spread = std[:, category_idx, season, coefficient]
                panel.plot(timesteps, values, color=f"C{coefficient}", label=f"theta_{coefficient}")
                panel.fill_between(
                    timesteps, values - spread, values + spread, color=f"C{coefficient}", alpha=0.12
                )
            panel.axhline(0.0, color="black", linestyle=":", linewidth=1)
            panel.axhline(1.0, color="black", linestyle=":", linewidth=1)
    axes[0, 0].legend(ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def action_clipping_figure(history: ActionStatistics) -> Figure:
    """Fraction of sampled coefficients clipped to the lower (solid) or upper (dashed) bound."""
    timesteps = np.asarray(history.timesteps)
    low = np.stack(history.clipped_low)
    high = np.stack(history.clipped_high)
    fig, axes = _category_season_grid(
        "Clipped actions: lower bound (solid) and upper bound (dashed)",
        "Fraction of samples",
        low.shape[1],
    )
    for category_idx in range(low.shape[1]):
        for season in range(2):
            panel = axes[category_idx, season]
            for coefficient in range(low.shape[3]):
                color = f"C{coefficient}"
                panel.plot(
                    timesteps,
                    low[:, category_idx, season, coefficient],
                    color=color,
                    label=f"theta_{coefficient}",
                )
                panel.plot(
                    timesteps,
                    high[:, category_idx, season, coefficient],
                    color=color,
                    linestyle="--",
                )
            panel.set_ylim(-0.02, 1.02)
    axes[0, 0].legend(ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ----------- POLICY CURVES ACROSS CHECKPOINTS -----------
def policy_probability_curves(coefficients: np.ndarray) -> np.ndarray:
    """BAR increment probability on OCCUPANCY_GRID for each category and season."""
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 3:
        raise ValueError("Coefficients must have shape (category, season, coefficient).")
    curves = np.empty((*coefficients.shape[:2], OCCUPANCY_GRID.size))
    for category_idx in range(coefficients.shape[0]):
        for season in range(coefficients.shape[1]):
            curves[category_idx, season] = BernsteinProbModel(
                coefficients[category_idx, season]
            ).predict_increment_probs(OCCUPANCY_GRID)
    return curves


class PolicyCurveAccumulator:
    """Average of the probability curves of the coefficients chosen on each evaluation step."""

    def __init__(self) -> None:
        self._sum: np.ndarray | None = None
        self._count = 0

    def add(self, coefficients: np.ndarray) -> None:
        curves = policy_probability_curves(coefficients)
        self._sum = curves if self._sum is None else self._sum + curves
        self._count += 1

    def mean(self) -> np.ndarray:
        if self._sum is None:
            raise ValueError("No policy coefficients recorded.")
        return self._sum / self._count


def policy_evolution_figure(curves_by_checkpoint: Mapping[int, np.ndarray]) -> Figure:
    """Mean increment-probability curves, one line per evaluation checkpoint."""
    checkpoints = sorted(curves_by_checkpoint)
    num_categories = curves_by_checkpoint[checkpoints[0]].shape[0]
    norm = Normalize(vmin=checkpoints[0], vmax=max(checkpoints[-1], checkpoints[0] + 1))
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(
        num_categories,
        2,
        figsize=(13, 3.4 * num_categories),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for category_idx in range(num_categories):
        for season, label in enumerate(SEASON_LABELS):
            panel = axes[category_idx, season]
            for checkpoint in checkpoints:
                panel.plot(
                    OCCUPANCY_GRID * 100,
                    curves_by_checkpoint[checkpoint][category_idx, season] * 100,
                    color=cmap(norm(checkpoint)),
                )
            panel.set_title(f"Category {category_idx + 1} | {label}")
            panel.grid(True, alpha=0.25)
        axes[category_idx, 0].set_ylabel("BAR increment probability (%)")
    for panel in axes[-1]:
        panel.set_xlabel("Night occupancy (%)")
    fig.suptitle("Evaluation policy: increment probability by occupancy, one line per checkpoint")
    fig.tight_layout(rect=(0, 0, 0.91, 0.97))
    colorbar_axis = fig.add_axes((0.93, 0.15, 0.015, 0.7))
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis, label="Timesteps")
    return fig


# ----------- EVALUATION RETURN VS BASELINES -----------
def eval_return_figure(
    returns_by_checkpoint: Mapping[int, Sequence[float]],
    baselines: Mapping[str, Sequence[float]] | None = None,
    title: str = "Evaluation return against baselines (same evaluation environments)",
) -> Figure:
    """Evaluation return per checkpoint (mean +/- std over episodes) against baseline returns."""
    checkpoints = sorted(returns_by_checkpoint)
    fig, panel = plt.subplots(figsize=(11, 6))
    means = np.asarray([np.mean(returns_by_checkpoint[step]) for step in checkpoints])
    stds = np.asarray([np.std(returns_by_checkpoint[step]) for step in checkpoints])
    panel.plot(checkpoints, means, color="C0", marker="o", label="PPO (deterministic)")
    panel.fill_between(checkpoints, means - stds, means + stds, color="C0", alpha=0.15)
    for step in checkpoints:
        episodes = np.asarray(returns_by_checkpoint[step], dtype=float)
        panel.scatter(np.full(episodes.size, step), episodes, color="C0", s=12, alpha=0.4)
    for index, (name, values) in enumerate((baselines or {}).items(), start=1):
        mean = float(np.mean(values))
        panel.axhline(
            mean,
            color=f"C{index}",
            linestyle="--",
            label=f"{name}: {mean:,.0f}",
        )
        if len(values) > 1:
            panel.axhspan(min(values), max(values), color=f"C{index}", alpha=0.08)
    panel.set(xlabel="Training timesteps", ylabel="Episode return (EUR)")
    panel.grid(True, alpha=0.25)
    panel.legend(fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ----------- VALUE FUNCTION VS REALIZED RETURN -----------
def discounted_returns(rewards: Sequence[float], gamma: float) -> np.ndarray:
    """Return-to-go G_t = r_t + gamma * G_{t+1} with a terminal episode end."""
    returns = np.zeros(len(rewards), dtype=float)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    return returns


def value_check_stats(values: np.ndarray, returns: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    returns = np.asarray(returns, dtype=float)
    if values.shape != returns.shape or values.ndim != 1 or values.size == 0:
        raise ValueError("Values and returns must be non-empty arrays of the same length.")
    error = values - returns
    return_var = float(np.var(returns))
    explained = 1.0 - float(np.var(error)) / return_var if return_var > 0 else float("nan")
    correlation = (
        float(np.corrcoef(values, returns)[0, 1])
        if np.std(values) > 0 and np.std(returns) > 0
        else float("nan")
    )
    return {
        "explained_variance": explained,
        "correlation": correlation,
        "mean_error": float(error.mean()),
    }


def value_check_figure(
    values: np.ndarray, returns: np.ndarray, stats: Mapping[str, float], title: str
) -> Figure:
    """Critic estimate V(s_t) against the discounted return-to-go on one evaluation episode."""
    days = np.arange(1, len(values) + 1)
    fig, (series_ax, scatter_ax) = plt.subplots(1, 2, figsize=(14, 5.5))
    series_ax.plot(days, returns, color="C1", label="Discounted return-to-go")
    series_ax.plot(days, values, color="C0", label="Critic V(s_t)")
    series_ax.set(xlabel="Request day", ylabel="EUR")
    series_ax.legend()
    limits = (min(values.min(), returns.min()), max(values.max(), returns.max()))
    scatter_ax.scatter(returns, values, s=12, alpha=0.6)
    scatter_ax.plot(limits, limits, color="black", linestyle=":", linewidth=1)
    scatter_ax.set(xlabel="Discounted return-to-go (EUR)", ylabel="Critic V(s_t) (EUR)")
    scatter_ax.text(
        0.04,
        0.96,
        f"Explained variance: {stats['explained_variance']:.3f}\n"
        f"Correlation: {stats['correlation']:.3f}\n"
        f"Mean error (V - G): {stats['mean_error']:,.0f} EUR",
        transform=scatter_ax.transAxes,
        va="top",
    )
    for panel in (series_ax, scatter_ax):
        panel.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
