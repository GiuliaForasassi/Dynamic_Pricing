from __future__ import annotations

import mlflow
import numpy as np
from matplotlib.figure import Figure

from pricing_advertising.requests.request import (
    ALTERNATIVE_TYPES,
    OFFER_TYPES,
    ORIGINAL_OFFER,
)

# ---------------- ARRAY LAYOUT ----------------
# Per offer type (rows follow OFFER_TYPES), the columns of the count arrays:
PROPOSED, PRICE_ACCEPTED, CONFIRMED, RECOVERY, REVENUE, REFERENCE_REVENUE = range(6)
N_COLUMNS = 6
# Per request-level counters:
REQUESTS, OUTSIDE_WINDOW, UNAVAILABLE = range(3)
N_REQUEST_COUNTERS = 3

ALTERNATIVE_ROWS = [OFFER_TYPES.index(name) for name in ALTERNATIVE_TYPES]
ORIGINAL_ROW = OFFER_TYPES.index(ORIGINAL_OFFER)

LABELS = {
    ORIGINAL_OFFER: "Original",
    "cat_1_to_2": "Category 1 → 2",
    "cat_1_to_3": "Category 1 → 3",
    "cat_2_to_3": "Category 2 → 3",
    "period_start_later_shorter": "Start +1 day, stay -1",
    "period_end_earlier": "Stay -1 (ends earlier)",
    "period_shift_earlier": "Shift -1 day",
    "period_shift_later": "Shift +1 day",
}
COLORS = {
    ORIGINAL_OFFER: "#7f7f7f",
    "cat_1_to_2": "#ff7f0e",
    "cat_1_to_3": "#d62728",
    "cat_2_to_3": "#e377c2",
    "period_start_later_shorter": "#1f77b4",
    "period_end_earlier": "#17becf",
    "period_shift_earlier": "#2ca02c",
    "period_shift_later": "#bcbd22",
}


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(np.shape(numerator), np.nan, dtype=float)
    np.divide(numerator, denominator, out=out, where=np.asarray(denominator) > 0)
    return out


def _label_bars(ax, bars, fmt: str = "{:.1f}") -> None:  # noqa: ANN001
    for bar in bars:
        height = bar.get_height()
        if np.isfinite(height):
            ax.annotate(
                fmt.format(height),
                (bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )


def _alt_axis(ax) -> list[str]:  # noqa: ANN001
    labels = [LABELS[name] for name in ALTERNATIVE_TYPES]
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    return labels


# ---------------- FIGURES ----------------
# All the inputs are means per completed episode.
def plot_by_checkin_day(
    split: str, by_checkin: np.ndarray, occupancy_pct: np.ndarray, num_episodes: int
) -> Figure:
    fig = Figure(figsize=(14, 5), layout="constrained")
    ax = fig.subplots()
    days = np.arange(1, by_checkin.shape[0] + 1)
    bottom = np.zeros(by_checkin.shape[0])
    for row in ALTERNATIVE_ROWS:
        name = OFFER_TYPES[row]
        values = by_checkin[:, row, PROPOSED]
        ax.bar(days, values, bottom=bottom, width=1.0, color=COLORS[name], label=LABELS[name])
        bottom += values
    ax.set(
        xlabel="Check-in day of the original request",
        ylabel="Alternatives proposed per episode",
        xlim=(0.5, by_checkin.shape[0] + 0.5),
    )
    ax.grid(True, axis="y", alpha=0.3)
    occupancy_ax = ax.twinx()
    occupancy_ax.plot(days, occupancy_pct, color="black", linewidth=1.2, label="Occupancy (%)")
    occupancy_ax.set(ylabel="Occupancy (%)", ylim=(0, 100))
    handles, labels = ax.get_legend_handles_labels()
    extra_handles, extra_labels = occupancy_ax.get_legend_handles_labels()
    ax.legend(handles + extra_handles, labels + extra_labels, loc="upper left", fontsize=8)
    fig.suptitle(
        f"Alternatives proposed by check-in day ({split}, mean of {num_episodes} episodes)"
    )
    return fig


def plot_by_type(split: str, totals: np.ndarray, num_episodes: int) -> Figure:
    fig = Figure(figsize=(10, 5), layout="constrained")
    ax = fig.subplots()
    proposed = totals[ALTERNATIVE_ROWS, PROPOSED]
    recovery = totals[ALTERNATIVE_ROWS, RECOVERY]
    colors = [COLORS[name] for name in ALTERNATIVE_TYPES]
    positions = np.arange(len(proposed))
    ax.bar(positions, proposed - recovery, color=colors, label="Before pricing")
    ax.bar(
        positions,
        recovery,
        bottom=proposed - recovery,
        color=colors,
        hatch="//",
        edgecolor="white",
        label="After a failed booking",
    )
    for position, value in zip(positions, proposed, strict=True):
        ax.annotate(
            f"{value:.1f}",
            (float(position), float(value)),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
        )
    _alt_axis(ax)
    ax.set_ylabel("Alternatives proposed per episode")
    ax.legend(fontsize=8)
    ax.set_title(f"Types of alternatives proposed ({split}, mean of {num_episodes} episodes)")
    return fig


def plot_funnel_by_type(split: str, totals: np.ndarray, num_episodes: int) -> Figure:
    fig = Figure(figsize=(15, 5.5), layout="constrained")
    rate_ax, count_ax = fig.subplots(1, 2)

    rows = [ORIGINAL_ROW, *ALTERNATIVE_ROWS]
    names = [OFFER_TYPES[row] for row in rows]
    proposed = totals[rows, PROPOSED]
    positions = np.arange(len(rows))
    width = 0.38
    rate_ax.bar(
        positions - width / 2,
        _ratio(totals[rows, PRICE_ACCEPTED], proposed),
        width,
        label="Price accepted / proposed",
    )
    confirmation = rate_ax.bar(
        positions + width / 2,
        _ratio(totals[rows, CONFIRMED], proposed),
        width,
        label="Confirmed / proposed",
    )
    _label_bars(rate_ax, confirmation, "{:.2f}")
    rate_ax.set_xticks(positions, [LABELS[name] for name in names], rotation=30, ha="right")
    rate_ax.set(ylabel="Rate", ylim=(0, 1.1), title="Rates by offer type")
    rate_ax.grid(True, axis="y", alpha=0.3)
    rate_ax.legend(fontsize=8)

    alt_positions = np.arange(len(ALTERNATIVE_ROWS))
    width = 0.27
    for shift, (column, label) in enumerate(
        ((PROPOSED, "Proposed"), (PRICE_ACCEPTED, "Price accepted"), (CONFIRMED, "Confirmed"))
    ):
        bars = count_ax.bar(
            alt_positions + (shift - 1) * width,
            totals[ALTERNATIVE_ROWS, column],
            width,
            label=label,
        )
        _label_bars(count_ax, bars, "{:.1f}")
    _alt_axis(count_ax)
    count_ax.set(ylabel="Alternatives per episode", title="Alternatives funnel")
    count_ax.legend(fontsize=8)
    fig.suptitle(f"Original offers vs alternatives ({split}, mean of {num_episodes} episodes)")
    return fig


def plot_request_funnel(
    split: str, totals: np.ndarray, requests: np.ndarray, num_episodes: int
) -> Figure:
    fig = Figure(figsize=(14, 5.5), layout="constrained")
    outcome_ax, funnel_ax = fig.subplots(1, 2)

    initial_alternatives = float(
        totals[ALTERNATIVE_ROWS, PROPOSED].sum() - totals[ALTERNATIVE_ROWS, RECOVERY].sum()
    )
    outcomes = [
        ("Outside booking\nwindow", requests[OUTSIDE_WINDOW], "#bbbbbb"),
        ("Lost: no availability,\nno alternative", requests[UNAVAILABLE], "#d62728"),
        ("Original offer", totals[ORIGINAL_ROW, PROPOSED], COLORS[ORIGINAL_OFFER]),
        ("Alternative offer", initial_alternatives, "#1f77b4"),
    ]
    bars = outcome_ax.bar(
        range(len(outcomes)), [value for _, value, _ in outcomes], color=[c for *_, c in outcomes]
    )
    _label_bars(outcome_ax, bars, "{:.1f}")
    outcome_ax.set_xticks(range(len(outcomes)), [label for label, *_ in outcomes], fontsize=8)
    outcome_ax.set(
        ylabel="Requests per episode",
        title=f"What happens to the {requests[REQUESTS]:.0f} requests of an episode",
    )
    outcome_ax.grid(True, axis="y", alpha=0.3)

    stages = (
        (PROPOSED, "Offers proposed"),
        (PRICE_ACCEPTED, "Price accepted"),
        (CONFIRMED, "Confirmed"),
    )
    original = np.asarray([totals[ORIGINAL_ROW, column] for column, _ in stages])
    alternatives = np.asarray([totals[ALTERNATIVE_ROWS, column].sum() for column, _ in stages])
    positions = np.arange(len(stages))
    funnel_ax.bar(positions, original, color=COLORS[ORIGINAL_OFFER], label="Original")
    funnel_ax.bar(positions, alternatives, bottom=original, color="#1f77b4", label="Alternatives")
    for position, (original_value, alternative_value) in enumerate(
        zip(original, alternatives, strict=True)
    ):
        funnel_ax.annotate(
            f"{original_value + alternative_value:.1f}",
            (position, original_value + alternative_value),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
        )
    funnel_ax.set_xticks(positions, [label for _, label in stages])
    funnel_ax.set(ylabel="Offers per episode", title="Offer funnel")
    funnel_ax.grid(True, axis="y", alpha=0.3)
    funnel_ax.legend(fontsize=8)
    fig.suptitle(f"Request outcomes ({split}, mean of {num_episodes} episodes)")
    return fig


def plot_revenue_impact(split: str, totals: np.ndarray, num_episodes: int) -> Figure:
    fig = Figure(figsize=(14, 5.5), layout="constrained")
    share_ax, ratio_ax = fig.subplots(1, 2)

    total_revenue = float(totals[:, REVENUE].sum())
    alternative_revenue = totals[ALTERNATIVE_ROWS, REVENUE]
    share = _ratio(alternative_revenue, np.full_like(alternative_revenue, total_revenue))
    bars = share_ax.bar(
        range(len(ALTERNATIVE_ROWS)),
        share * 100,
        color=[COLORS[name] for name in ALTERNATIVE_TYPES],
    )
    _label_bars(share_ax, bars, "{:.2f}%")
    _alt_axis(share_ax)
    overall_share = alternative_revenue.sum() / total_revenue * 100 if total_revenue > 0 else 0.0
    share_ax.set(
        ylabel="Share of total revenue (%)",
        title=f"Revenue from alternatives: {overall_share:.2f}% of {total_revenue:,.0f}",
    )

    ratio = _ratio(alternative_revenue, totals[ALTERNATIVE_ROWS, REFERENCE_REVENUE])
    bars = ratio_ax.bar(
        range(len(ALTERNATIVE_ROWS)),
        ratio,
        color=[COLORS[name] for name in ALTERNATIVE_TYPES],
    )
    _label_bars(ratio_ax, bars, "{:.2f}")
    ratio_ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    _alt_axis(ratio_ax)
    ratio_ax.set(
        ylabel="Price paid / price of the original request",
        title="Price of a confirmed alternative vs the original request",
    )
    fig.suptitle(f"Revenue impact of alternatives ({split}, mean of {num_episodes} episodes)")
    return fig


# ---------------- LOGGING ----------------
def log_alternative_figures(
    split: str,
    *,
    totals: np.ndarray,
    by_checkin: np.ndarray,
    occupancy_pct: np.ndarray,
    requests: np.ndarray,
    num_episodes: int,
) -> None:
    figures = {
        "by_checkin_day": plot_by_checkin_day(split, by_checkin, occupancy_pct, num_episodes),
        "by_type": plot_by_type(split, totals, num_episodes),
        "funnel_by_type": plot_funnel_by_type(split, totals, num_episodes),
        "request_funnel": plot_request_funnel(split, totals, requests, num_episodes),
        "revenue_impact": plot_revenue_impact(split, totals, num_episodes),
    }
    for name, fig in figures.items():
        mlflow.log_figure(fig, f"alternatives/{split}/{name}.png")
