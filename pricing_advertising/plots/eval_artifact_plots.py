from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from matplotlib.ticker import MaxNLocator

from pricing_advertising.pricing_model.bernstein_model import BernsteinProbModel
from pricing_advertising.utils.constants import HIGH_SEASON


# ----------- HELPER FUNCTION -----------
# Utility functions for evaluation artifact plots: division function with safe handling of zero denominators
def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _request_price_series(
    steps: list[dict[str, Any]],
    request_date: date,
    start: date,
    end: date,
    *,
    confirmed_only: bool = False,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    prices_by_category: dict[int, dict[int, float]] = {}
    for step in steps:
        if step["request_day"] != request_date.isoformat():
            continue
        for request in step["requests"]:
            if not request["valid"] or (confirmed_only and not request["confirmed"]):
                continue
            check_in = date.fromisoformat(request["check_in"])
            first_night = max(0, (request_date - check_in).days)
            stop_night = min(int(request["days_stay"]), (end - check_in).days + 1)
            if first_night >= stop_night:
                continue
            days = (check_in - start).days + 1 + np.arange(first_night, stop_night)
            prices = np.asarray(request["daily_prices"], dtype=float)[first_night:stop_night]
            category = int(request["category"])
            nightly_prices = prices_by_category.setdefault(category, {})
            for day, price in zip(days, prices, strict=True):
                day_index = int(day)
                if day_index in nightly_prices and nightly_prices[day_index] != price:
                    raise ValueError(
                        f"Conflicting prices for category {category}, stay day {day}, "
                        f"requests received on {request_date}."
                    )
                nightly_prices[day_index] = float(price)
    series = []
    for category, nightly_prices in sorted(prices_by_category.items()):
        days = np.asarray(sorted(nightly_prices), dtype=int)
        prices = np.asarray([nightly_prices[int(day)] for day in days], dtype=float)
        series.append((category, days, prices))
    return series


# ----------- LOG PROPOSED PRICE BY ADVANCE -----------
# Function to log prices offered on a given request date by stay-day index.
def log_proposed_price_by_advance(artifact: dict[str, Any], request_dates: list[date]) -> None:
    # Extract the steps and metadata from the artifact dictionary.
    steps = artifact["steps"]
    # Determine the start and end dates of the simulation.
    start = date.fromisoformat(steps[0]["start_simulation"])
    end = date.fromisoformat(steps[0]["end_simulation"])
    # Extract the checkpoint, environment index, and episode ID from the artifact dictionary.
    checkpoint = int(artifact["checkpoint_timesteps"])
    env_idx = int(artifact["env_index"])
    episode_id = int(artifact["episode_id"])

    for request_date in sorted(set(request_dates)):
        if not start <= request_date <= end:
            raise ValueError("Request date is outside the simulation period.")
        series = _request_price_series(steps, request_date, start, end)
        first_day = (request_date - start).days + 1
        last_day = (end - start).days + 1
        # Create subplots for each category to visualize proposed prices by advance.
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)
        try:
            for category, ax in enumerate(axes, start=1):
                selected = [item for item in series if item[0] == category]
                for _, days, prices in selected:
                    ax.plot(days, prices, marker=".", markersize=3, color=f"C{category - 1}")
                ax.set(title=f"Category {category}", ylabel="Proposed nightly price")
                ax.grid(True, alpha=0.3)
                if not selected:
                    ax.plot([], [], color=f"C{category - 1}")
                    ax.text(0.5, 0.5, "No observed offers", transform=ax.transAxes, ha="center")
            axes[-1].set(
                xlabel="Stay day (1 = simulation start)",
                xlim=(first_day, max(first_day + 1, last_day)),
            )
            axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
            fig.suptitle(
                f"Requests received on {request_date} only | Timesteps: {checkpoint}\n"
                f"Env: {env_idx} | Episode: {episode_id}"
            )
            fig.text(
                0.08,
                0.025,
                "One line per category: actual prices for observed stay nights.\n",
                fontsize=9,
                va="bottom",
            )
            fig.tight_layout(rect=(0, 0.12, 1, 0.95))
            mlflow.log_figure(
                fig,
                f"proposed_price_by_advance/{request_date}/"
                f"{checkpoint:08d}_timesteps/env_{env_idx:02d}/episode_{episode_id:05d}.png",
            )
        finally:
            plt.close(fig)


def _bar_increment_probabilities(
    snapshot: dict[str, Any], occupancy_before: np.ndarray, rooms: np.ndarray
) -> np.ndarray:
    probabilities = np.full(occupancy_before.shape, np.nan, dtype=float)
    for category_idx, theta in enumerate(snapshot.get("policy_coefficients", [])):
        if rooms[category_idx] > 0:
            coefficients = np.asarray(theta, dtype=float)
            if coefficients.ndim == 1:
                probabilities[category_idx] = BernsteinProbModel(
                    coefficients[::-1]
                ).predict_increment_probs(occupancy_before[category_idx])
                continue
            if "seasonality_by_day" in snapshot:
                seasons = np.asarray(snapshot["seasonality_by_day"], dtype=int)
            else:
                start = date.fromisoformat(snapshot["snapshot_start_day"])
                seasons = np.asarray(
                    [
                        int((start + timedelta(days=offset)).month in HIGH_SEASON)
                        for offset in range(occupancy_before.shape[1])
                    ]
                )
            for season in range(2):
                mask = seasons == season
                probabilities[category_idx, mask] = BernsteinProbModel(
                    coefficients[season]
                ).predict_increment_probs(occupancy_before[category_idx, mask])
    return probabilities


def log_bar_by_advance(artifact: dict[str, Any], *, show_bookings: bool = True) -> None:
    steps = artifact["steps"]
    checkpoint = int(artifact["checkpoint_timesteps"])
    env_idx = int(artifact["env_index"])
    episode_id = int(artifact["episode_id"])
    snapshots = {step["snapshot_start_day"]: step for step in steps if "snapshot_start_day" in step}

    for request_day in sorted(set(artifact["snapshot_dates"])):
        snapshot = snapshots[request_day]
        if "max_bar_level" not in snapshot:
            continue
        metrics = build_snapshot_metrics(steps, snapshot)
        advance = np.arange(len(metrics["day"]))
        rooms = np.asarray(snapshot["rooms_by_category"], dtype=float)
        levels = np.asarray(snapshot["bar_levels_by_category"])
        levels_before = np.asarray(snapshot["bar_levels_before_update"])
        occupancy_after = np.asarray(snapshot["occupancy_by_category"], dtype=float)
        daily_bookings = np.zeros_like(occupancy_after)
        points = metrics["booked_price_points"]
        np.add.at(
            daily_bookings,
            (points[:, 1].astype(int) - 1, points[:, 0].astype(int) - metrics["day"][0]),
            1,
        )
        occupancy_before = _ratio(occupancy_after - daily_bookings, rooms[:, None])
        booked_pct = _ratio(daily_bookings, rooms[:, None]) * 100
        probabilities = _bar_increment_probabilities(snapshot, occupancy_before, rooms)
        max_level = int(snapshot["max_bar_level"])
        eligible = (levels_before < max_level) & (advance[None, :] > 0)
        if "bar_update_end_day" in snapshot:
            stop_advance = (
                date.fromisoformat(snapshot["bar_update_end_day"]) - date.fromisoformat(request_day)
            ).days
            eligible &= advance[None, :] < stop_advance
        effective_probabilities = np.where(eligible, probabilities, 0.0)
        horizon = int(advance[-1])

        for folder, view, limit in (
            ("bar_by_advance", "All future stay nights", max(1, horizon)),
            ("bar_by_advance_next_60_days", "Next 60 days", 60),
        ):
            fig, (bar_ax, occupancy_ax, probability_ax) = plt.subplots(
                3, 1, figsize=(13, 10), sharex=True
            )
            try:
                for category_idx in range(len(rooms)):
                    color = f"C{category_idx}"
                    label = f"Category {category_idx + 1}"
                    bar_ax.step(
                        advance,
                        levels[category_idx],
                        where="mid",
                        color=color,
                        label=f"{label}: after update",
                    )
                    bar_ax.step(
                        advance,
                        levels_before[category_idx],
                        where="mid",
                        color=color,
                        linestyle="--",
                        label=f"{label}: before update",
                    )
                    occupancy_ax.step(
                        advance,
                        booked_pct[category_idx],
                        where="mid",
                        color=color,
                        marker="^" if show_bookings else None,
                        markevery=np.flatnonzero(daily_bookings[category_idx] > 0).tolist(),
                        markersize=3,
                    )
                    probability_ax.step(
                        advance,
                        effective_probabilities[category_idx] * 100,
                        where="mid",
                        color=color,
                        label=f"{label}: effective",
                    )
                    probability_ax.step(
                        advance,
                        probabilities[category_idx] * 100,
                        where="mid",
                        color=color,
                        linestyle=":",
                        alpha=0.65,
                        label=f"{label}: model",
                    )

                bar_ax.axhline(max_level, linestyle="--", color="black", linewidth=1)
                bar_ax.set(
                    ylabel="BAR level after update",
                    ylim=(-0.5, max_level + max(2, max_level * 0.08)),
                    title=f"Maximum BAR: {max_level}",
                )
                bar_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                occupancy_ax.set(ylabel="Occupancy added by today's bookings (%)", ylim=(-2, 106))
                probability_ax.set(
                    xlabel="Days from request date to stay night",
                    ylabel="BAR increment probability (%)",
                    ylim=(-2, 106),
                    xlim=(-0.5, limit + 0.5),
                )
                probability_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                if not np.isfinite(probabilities).any():
                    probability_ax.text(
                        0.5,
                        0.5,
                        "Model probabilities unavailable",
                        transform=probability_ax.transAxes,
                        ha="center",
                    )
                for panel in (bar_ax, occupancy_ax, probability_ax):
                    panel.grid(True, alpha=0.25)
                    if limit > horizon:
                        panel.axvspan(horizon + 0.5, limit + 0.5, color="0.92", zorder=0)
                bar_ax.legend(ncol=len(rooms), fontsize=8)
                probability_ax.legend(ncol=len(rooms), fontsize=8)
                fig.suptitle(
                    f"Decision on {request_day} | Timesteps: {checkpoint}\n"
                    f"Env: {env_idx} | Episode: {episode_id} | {view}"
                )
                fig.text(
                    0.08,
                    0.025,
                    "Occupancy shows only today's bookings; probabilities use hotel occupancy before today's bookings.\n",
                    fontsize=9,
                )
                fig.tight_layout(rect=(0, 0.07, 1, 0.93))
                mlflow.log_figure(
                    fig,
                    f"{folder}/{request_day}/{checkpoint:08d}_timesteps/"
                    f"env_{env_idx:02d}/episode_{episode_id:05d}.png",
                )
            finally:
                plt.close(fig)


# ------- BUILD SNAPSHOT METRICS -------
# Build snapshot metrics from requests received on the snapshot date.
def build_snapshot_metrics(
    steps: list[dict[str, Any]],
    snapshot: dict[str, Any],
) -> dict[str, np.ndarray]:
    # Take the snapshot start day as the cutoff for building snapshot metrics.
    cutoff = date.fromisoformat(snapshot["snapshot_start_day"])
    start = date.fromisoformat(snapshot["start_simulation"])
    end = date.fromisoformat(snapshot["end_simulation"])

    if not start <= cutoff <= end:
        raise ValueError("Cutoff date is outside the simulation period.")

    # Number of days from the cutoff to the end of the simulation.
    size = (end - cutoff).days + 1
    # Number of the first day of the snapshot relative to the start of the simulation.
    first_day = (cutoff - start).days + 1

    # Initialize arrays to store metrics for each day in the snapshot period.
    requests = np.zeros(size, dtype=np.int64)
    confirmed_requests = np.zeros(size, dtype=np.int64)
    offers = np.zeros(size, dtype=np.int64)
    price_accepted = np.zeros(size, dtype=np.int64)
    confirmed = np.zeros(size, dtype=np.int64)
    revenue = np.zeros(size, dtype=float)
    booked_price_points: list[tuple[int, int, float]] = []

    # Iterate over each step to accumulate metrics for the snapshot period.
    for step in steps:
        request_day = date.fromisoformat(step["request_day"])

        # Exclude requests received on other days.
        if request_day != cutoff:
            continue

        for request in step["requests"]:
            if date.fromisoformat(request["request_date"]) != request_day:
                raise ValueError("Request associated with the wrong day.")

            is_valid = bool(request["valid"])
            is_accepted = bool(request["price_accepted"])
            is_confirmed = bool(request["confirmed"])

            if is_accepted and not is_valid:
                raise ValueError("Price accepted without an offer.")
            if is_confirmed and not is_accepted:
                raise ValueError("Confirmation without price acceptance.")

            # Take the original request, if it contains "requested" field means that it was modified
            original = request.get("requested", request)
            if date.fromisoformat(original["request_date"]) != request_day:
                raise ValueError("Original request associated with the wrong day.")
            # Compute the start and stop indices for the original request within the snapshot period.
            original_offset = (date.fromisoformat(original["check_in"]) - cutoff).days
            original_start = max(0, original_offset)
            original_stop = min(size, original_offset + int(original["days_stay"]))
            if original_start < original_stop:
                requests[original_start:original_stop] += 1
                confirmed_requests[original_start:original_stop] += int(is_confirmed)

            # Compute the start and stop indices for the actual request within the snapshot period.
            stay = int(request["days_stay"])
            check_in = date.fromisoformat(request["check_in"])
            prices = request["daily_prices"]

            if is_valid:
                if prices is None or len(prices) != stay:
                    raise ValueError("Daily prices not aligned with the stay.")

            if is_confirmed:
                revenue[0] += float(np.sum(prices))

            # Compute the intersection of the stay with the snapshot period.
            offset = (check_in - cutoff).days
            first_night = max(0, -offset)
            stop_night = min(stay, size - offset)

            for night in range(first_night, stop_night):
                idx = offset + night

                offers[idx] += int(is_valid)
                price_accepted[idx] += int(is_accepted)
                confirmed[idx] += int(is_confirmed)

                if is_confirmed:
                    booked_price_points.append(
                        (first_day + idx, int(request["category"]), float(prices[night]))
                    )

    # Convert the lists to numpy arrays for efficient numerical operations.
    occupancy = np.asarray(snapshot["occupancy_by_category"], dtype=float)
    bar_prices = np.asarray(snapshot["bar_prices_by_category"], dtype=float)
    rooms = np.asarray(snapshot["rooms_by_category"], dtype=float)

    expected_shape = (len(rooms), size)
    if occupancy.shape != expected_shape or bar_prices.shape != expected_shape:
        raise ValueError("Snapshot dimensions are not consistent with the dates.")

    if rooms.sum() <= 0:
        raise ValueError("Total hotel capacity is not positive.")

    # Return a dictionary containing all the computed metrics for the snapshot period.
    return {
        "day": np.arange(first_day, first_day + size),
        "requests": requests,  # Total number of requests received for each day in the snapshot period.
        "confirmed": confirmed,  # Total number of confirmed requests for each day in the snapshot period.
        "booked_revenue": revenue,
        "confirmation_rate": _ratio(
            confirmed_requests, requests
        ),  # Ratio of confirmed requests to total requests for each day in the snapshot period.
        "price_acceptance_rate": _ratio(
            price_accepted, offers
        ),  # Ratio of accepted prices to total offers for each day in the snapshot period.
        "occupancy_pct": confirmed
        / rooms.sum()
        * 100.0,  # Occupancy percentage for each day in the snapshot period.
        "bar_prices": bar_prices,  # BAR (Best Available Rate) prices for each day in the snapshot period.
        "booked_price_points": np.asarray(booked_price_points, dtype=float).reshape(
            -1, 3
        ),  # List of booked price points as (day, category, price) tuples.
    }


# ----------- LOG EVAL ARTIFACT PLOTS -----------
def log_eval_artifact_plots(artifact: dict[str, Any]) -> None:
    # Take the steps from the artifact and ensure they are not empty.
    steps = artifact["steps"]
    if not steps:
        raise ValueError("Empty evaluation episode.")

    # Extract the start and end dates of the simulation from the first step.
    start = date.fromisoformat(steps[0]["start_simulation"])
    end = date.fromisoformat(steps[0]["end_simulation"])

    # Continuity check: no missing or duplicate days.
    for index, step in enumerate(steps):
        expected_day = start + timedelta(days=index)
        if step["request_day"] != expected_day.isoformat() or int(step["day_index"]) != index:
            raise ValueError("Episode day sequence is not continuous.")

    # End_simulation is included among the decision days.
    expected_steps = (end - start).days + 1

    if len(steps) != expected_steps or steps[-1]["request_day"] != end.isoformat():
        raise ValueError("Unexpected last step or incomplete episode.")

    # Extract snapshots from the steps.
    snapshots = [step for step in steps if "snapshot_start_day" in step]
    expected_dates = set(artifact["snapshot_dates"])
    actual_dates = {step["snapshot_start_day"] for step in snapshots}

    # Ensure that the actual snapshot dates match the expected snapshot dates.
    if actual_dates != expected_dates:
        raise ValueError(
            f"Missing or unexpected snapshots: "
            f"expected={sorted(expected_dates)}, found={sorted(actual_dates)}"
        )

    # Extract checkpoint, environment index, and episode ID from the artifact.
    checkpoint = int(artifact["checkpoint_timesteps"])
    env_idx = int(artifact["env_index"])
    episode_id = int(artifact["episode_id"])

    # - Log proposed price by advance for specific dates.
    log_proposed_price_by_advance(
        artifact,
        [date.fromisoformat(value) for value in sorted(expected_dates)],
    )
    log_bar_by_advance(artifact)

    # Specifications for different types of plots: key, folder, Y-axis label, Y-axis limits.
    specifications = [
        (
            "booked_price_points",
            "booked_price_plots",
            "Booked nightly price (offered nights)",
            None,
        ),
        (
            "requests",
            "request_plots",
            "Requests covering the originally requested night",
            None,
        ),
        (
            "confirmation_rate",
            "confirmation_rate_plots",
            "Confirmation rate (originally requested nights)\nConfirmed requests / all requests",
            (0, 1),
        ),
        (
            "price_acceptance_rate",
            "price_acceptance_plots",
            "Price acceptance rate (offered nights)\nPrice-accepted offers / all offers",
            (0, 1),
        ),
        (
            "occupancy_pct",
            "occupancy_plots",
            "Occupancy added by today's bookings (%) (confirmed offered nights)",
            (0, 100),
        ),
        (
            "bar_prices",
            "bar_price_plots",
            "Current BAR nightly price",
            None,
        ),
    ]

    # - Iterate over each snapshot to generate plots for policy diagnostics and specified metrics.
    for snapshot in snapshots:
        cutoff = snapshot["snapshot_start_day"]
        metrics = build_snapshot_metrics(steps, snapshot)
        x = metrics["day"]

        coefficients = np.asarray(snapshot["policy_coefficients"])
        before = np.asarray(snapshot["bar_levels_before_update"])
        after = np.asarray(snapshot["bar_levels_by_category"])
        occupancy_grid = np.linspace(0.0, 1.0, 101)
        fig, axes = plt.subplots(len(before), 3, figsize=(18, 4 * len(before)), squeeze=False)
        try:
            for category_idx, panels in enumerate(axes):
                coefficient_ax, bar_ax, probability_ax = panels
                seasonal = coefficients.ndim == 3
                curves = (
                    coefficients[category_idx] if seasonal else coefficients[category_idx][None, :]
                )
                labels = ("Low season", "High season") if seasonal else ("All seasons (legacy)",)
                for theta, label in zip(curves, labels, strict=True):
                    coefficient_ax.plot(np.arange(theta.size), theta, marker=".", label=label)
                    model = BernsteinProbModel(theta if seasonal else theta[::-1])
                    probability_ax.plot(
                        occupancy_grid * 100,
                        model.predict_increment_probs(occupancy_grid),
                        label=label,
                    )
                coefficient_ax.legend()
                probability_ax.legend()
                coefficient_ax.set(
                    xlabel="Coefficient index",
                    ylabel="Coefficient for today's decision",
                    ylim=(0, 1),
                )
                coefficient_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                bar_ax.plot(x, before[category_idx], "--", label="Before")
                bar_ax.plot(x, after[category_idx], label="After")
                bar_ax.set(xlabel="Stay day (1 = start)", ylabel="BAR level")
                bar_ax.legend()
                probability_ax.set(
                    xlabel="Occupancy (%)", ylabel="Increment probability", ylim=(0, 1)
                )
                for panel in panels:
                    panel.set_title(f"Category {category_idx + 1}")
                    panel.grid(True, alpha=0.3)
            fig.suptitle(
                f"Decision on {cutoff} | Timesteps: {checkpoint} | Env: {env_idx} | Episode: {episode_id}"
            )
            fig.tight_layout()
            mlflow.log_figure(
                fig,
                f"policy_diagnostics/{cutoff}/{checkpoint:08d}_timesteps/"
                f"env_{env_idx:02d}/episode_{episode_id:05d}.png",
            )
        finally:
            plt.close(fig)

        for key, folder, ylabel, ylim in specifications:
            is_booked = key == "booked_price_points"
            num_panels = len(snapshot["rooms_by_category"]) if is_booked else 1
            fig, axes = plt.subplots(
                num_panels,
                1,
                figsize=(12, 3 * num_panels if is_booked else 5),
                sharex=True,
                sharey=True,
                squeeze=False,
                layout="constrained" if is_booked else None,
            )

            try:
                if is_booked:
                    series = _request_price_series(
                        steps, date.fromisoformat(cutoff), start, end, confirmed_only=True
                    )
                    for category, panel in enumerate(axes.flat, start=1):
                        selected = [item for item in series if item[0] == category]
                        for _, days, prices in selected:
                            panel.plot(
                                days, prices, marker=".", markersize=3, color=f"C{category - 1}"
                            )
                        panel.set_title(f"Category {category}")
                        if not selected:
                            panel.plot([], [], color=f"C{category - 1}")
                            panel.text(
                                0.5,
                                0.5,
                                "No confirmed bookings",
                                transform=panel.transAxes,
                                ha="center",
                            )
                else:
                    ax = axes[0, 0]
                    if key == "bar_prices":
                        for category, prices in enumerate(metrics[key], start=1):
                            ax.plot(x, prices, label=f"Category {category}")
                        ax.legend()
                    else:
                        values = metrics[key]
                        ax.plot(x, values, marker=".", markersize=2)
                        if not np.isfinite(values).any():
                            ax.text(
                                0.5,
                                0.5,
                                "No observations",
                                transform=ax.transAxes,
                                ha="center",
                            )

                for panel in axes.flat:
                    panel.set_ylabel(ylabel)
                    panel.set_xlim(float(x[0]) - 0.5, float(x[-1]) + 0.5)
                    if is_booked:
                        panel.set_xlim(float(x[0]), max(float(x[0]) + 1, float(x[-1])))
                        panel.xaxis.set_major_locator(MaxNLocator(integer=True))
                    if ylim is not None:
                        panel.set_ylim(*ylim)
                    panel.grid(True, alpha=0.3)

                axes[-1, 0].set_xlabel("Stay day (1 = simulation start)")
                fig.suptitle(
                    f"{ylabel}\n"
                    f"{'BAR state on' if key == 'bar_prices' else 'Requests received only on'} {cutoff} | "
                    f"Training timesteps: {checkpoint} | "
                    f"Env: {env_idx} | Episode: {episode_id}"
                )
                if not is_booked:
                    fig.tight_layout()
                mlflow.log_figure(
                    fig,
                    f"{folder}/{cutoff}/{checkpoint:08d}_timesteps/"
                    f"env_{env_idx:02d}/episode_{episode_id:05d}.png",
                )
            finally:
                plt.close(fig)
