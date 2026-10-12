from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pytest
from matplotlib.colors import to_rgba

from pricing_advertising.plots import eval_artifact_plots as plots
from pricing_advertising.plots.eval_artifact_plots import (
    _proposed_price_series,
    build_snapshot_metrics,
)


@pytest.mark.parametrize("stored_seasons", [False, True])
def test_six_policy_curves_and_stay_season_probabilities(monkeypatch, stored_seasons):
    start = date(2025, 6, 30)
    coefficients = np.zeros((3, 2, 4))
    scales = np.array([1.0, 0.5, 0.25])
    coefficients[:, 0, 0] = scales
    coefficients[:, 1, 3] = scales
    steps = [
        {
            "start_simulation": str(start),
            "end_simulation": "2025-07-02",
            "request_day": str(start + timedelta(days=offset)),
            "day_index": offset,
            "requests": [],
            "policy_coefficients": coefficients.tolist(),
        }
        for offset in range(3)
    ]
    snapshot = steps[0]
    snapshot.update(
        snapshot_start_day=str(start),
        rooms_by_category=[10, 10, 10],
        occupancy_by_category=[[2, 4, 6]] * 3,
        bar_prices_by_category=[[100] * 3] * 3,
        bar_levels_before_update=[[1] * 3] * 3,
        bar_levels_by_category=[[1] * 3] * 3,
        max_bar_level=10,
    )
    seasons = np.array([1, 0, 1] if stored_seasons else [0, 1, 1])
    if stored_seasons:
        snapshot["seasonality_by_day"] = seasons.tolist()
    snapshot["bar_update_end_day"] = "2025-07-02"
    figures = {}
    monkeypatch.setattr(
        plots.mlflow, "log_figure", lambda fig, path: figures.update({path.split("/")[0]: fig})
    )
    plots.log_eval_artifact_plots(
        {
            "checkpoint_timesteps": 10,
            "env_index": 0,
            "episode_id": 0,
            "snapshot_dates": [str(start)],
            "steps": steps,
        }
    )
    probability_panel = figures["bar_by_advance"].axes[2]
    occupancy = np.array([0.2, 0.4, 0.6])
    for category_idx, scale in enumerate(scales):
        expected = scale * np.where(seasons == 0, 0.25, occupancy**3 / 4)
        np.testing.assert_allclose(
            probability_panel.lines[2 * category_idx + 1].get_ydata(), 100 * expected
        )
        expected[0] = 0
        expected[2] = 0
        np.testing.assert_allclose(
            probability_panel.lines[2 * category_idx].get_ydata(), 100 * expected
        )
        panels = figures["policy_diagnostics"].axes[3 * category_idx : 3 * category_idx + 3]
        assert panels[1].get_ylim() == (0, 10 * 1.05)
        assert any(line.get_label() == "Max level (10)" for line in panels[1].lines)
        assert len(panels[0].containers) == 2
        assert len(panels[2].lines) == 2
        for season, label in enumerate(("Low season", "High season")):
            container = panels[0].containers[season]
            assert container.get_label() == label
            np.testing.assert_array_equal(
                [bar.get_height() for bar in container], coefficients[category_idx, season]
            )
            line = panels[2].lines[season]
            assert line.get_label() == label
            grid = line.get_xdata() / 100
            np.testing.assert_allclose(
                line.get_ydata(), scale * (np.full_like(grid, 0.25) if season == 0 else grid**3 / 4)
            )


@pytest.mark.parametrize("with_history", [False, True])
@pytest.mark.parametrize(
    ("start", "end", "with_snapshots"),
    [
        (date(2024, 2, 28), date(2024, 2, 29), True),
        (date(2026, 12, 31), date(2027, 1, 1), True),
        (date(2026, 9, 29), date(2026, 9, 29), True),
        (date(2026, 9, 29), date(2026, 9, 30), False),
    ],
)
def test_plot_dates_follow_snapshots_and_labels_distinguish_nights(
    monkeypatch, start, end, with_snapshots, with_history
):
    snapshot_dates = sorted({start.isoformat(), end.isoformat()}) if with_snapshots else []
    steps = []
    for index in range((end - start).days + 1):
        day = start + timedelta(days=index)
        step = {
            "start_simulation": start.isoformat(),
            "end_simulation": end.isoformat(),
            "request_day": day.isoformat(),
            "day_index": index,
            "requests": [],
            "policy_coefficients": [[0.5] * 4] * 3,
        }
        if with_history:
            step["max_bar_level"] = 3
            step["bar_history"] = {
                stay_day: {
                    "levels_before": [0, 0, 0],
                    "levels_after": [0, 0, 0],
                    "occupancy_before": [0, 0, 0],
                    "occupancy_after": [0, 0, 0],
                }
                for stay_day in snapshot_dates
                if stay_day >= day.isoformat()
            }
        if day.isoformat() in snapshot_dates:
            grid = [[0] * ((end - day).days + 1) for _ in range(3)]
            step.update(
                snapshot_start_day=day.isoformat(),
                rooms_by_category=[2, 2, 2],
                occupancy_by_category=grid,
                bar_prices_by_category=grid,
                bar_levels_before_update=grid,
                bar_levels_by_category=grid,
            )
        steps.append(step)

    logged = {}

    def capture_figure(fig, path):
        logged[path] = fig.axes[0].get_ylabel()

    monkeypatch.setattr(plots.mlflow, "log_figure", capture_figure)
    plots.log_eval_artifact_plots(
        {
            "checkpoint_timesteps": 800,
            "env_index": 0,
            "episode_id": 0,
            "snapshot_dates": snapshot_dates,
            "steps": steps,
        }
    )

    proposed_paths = [path for path in logged if path.startswith("proposed_price_by_advance/")]
    assert [path.split("/")[1] for path in proposed_paths] == snapshot_dates
    assert len(logged) == (7 + int(with_history)) * len(snapshot_dates)
    assert not any(path.startswith("request_plots/") for path in logged)
    for path, ylabel in logged.items():
        folder = path.split("/")[0]
        if folder == "confirmation_rate_plots":
            assert "originally requested night" in ylabel
        elif folder in {"booked_price_plots", "price_acceptance_plots", "occupancy_plots"}:
            assert "offered nights" in ylabel


@pytest.mark.parametrize("num_days", [1, 3, 61, 63])
def test_bar_snapshot_uses_current_policy_and_daily_bookings(
    monkeypatch: pytest.MonkeyPatch, num_days: int
) -> None:
    request_day = date(2025, 2, 28)
    end = request_day + timedelta(days=num_days - 1)
    advance = np.arange(num_days)
    before = np.array([[1] * num_days, [3] * num_days, [0] * num_days])
    after = before.copy()
    after[0, 1:] = 2
    snapshot = {
        "start_simulation": "2025-01-01",
        "end_simulation": end.isoformat(),
        "request_day": request_day.isoformat(),
        "snapshot_start_day": request_day.isoformat(),
        "rooms_by_category": [10, 10, 0],
        "occupancy_by_category": [[6] * num_days, [10] * num_days, [0] * num_days],
        "bar_prices_by_category": [[100] * num_days] * 3,
        "bar_levels_before_update": before.tolist(),
        "bar_levels_by_category": after.tolist(),
        "max_bar_level": 3,
        "policy_coefficients": [[1.0] * 4, [0.0] * 4, [0.0] * 4],
        "requests": [make_request(str(request_day), str(request_day), num_days, [100] * num_days)],
    }
    artifact = {
        "checkpoint_timesteps": 800,
        "env_index": 1,
        "episode_id": 2,
        "snapshot_dates": [request_day.isoformat()],
        "steps": [
            {"request_day": "2025-02-20", "policy_coefficients": [[0.0] * 4] * 3},
            snapshot,
            {"request_day": (end + timedelta(days=1)).isoformat()},
        ],
    }
    original = deepcopy(artifact)
    logged = []
    monkeypatch.setattr(plots.mlflow, "log_figure", lambda fig, path: logged.append((fig, path)))

    plots.log_bar_by_advance(artifact)

    assert len(logged) == 1
    assert (
        logged[0][1] == f"bar_by_advance/{request_day}/00000800_timesteps/env_01/episode_00002.png"
    )
    expected_model = np.full(num_days, 62.5)
    for fig, path in logged:
        bar_ax, increase_ax, probability_ax = fig.axes
        assert len(bar_ax.lines) == 4
        assert [text.get_text() for text in bar_ax.get_legend().get_texts()] == [
            "Category 1",
            "Category 2",
            "Category 3",
        ]
        np.testing.assert_array_equal(bar_ax.lines[-1].get_ydata(), [3, 3])
        for category_idx in range(3):
            np.testing.assert_array_equal(bar_ax.lines[category_idx].get_xdata(), advance)
            np.testing.assert_array_equal(
                bar_ax.lines[category_idx].get_ydata(), after[category_idx]
            )
            assert bar_ax.lines[category_idx].get_label() == f"Category {category_idx + 1}"
            assert bar_ax.lines[category_idx].get_linestyle() == "-"
        np.testing.assert_array_equal(
            increase_ax.collections[0].get_offsets(),
            np.column_stack((advance[1:], np.ones(num_days - 1))),
        )
        for collection in increase_ax.collections[1:]:
            assert collection.get_offsets().shape == (0, 2)
        assert increase_ax.get_title() == "Actual BAR increases (+1)"
        assert [tick.get_text() for tick in increase_ax.get_yticklabels()] == [
            "Category 1",
            "Category 2",
            "Category 3",
        ]
        assert increase_ax.yaxis_inverted()
        assert bool(increase_ax.texts) == (num_days == 1)
        np.testing.assert_allclose(
            probability_ax.lines[0].get_ydata(), np.where(advance > 0, expected_model, 0)
        )
        np.testing.assert_allclose(probability_ax.lines[1].get_ydata(), expected_model)
        np.testing.assert_array_equal(probability_ax.lines[2].get_ydata(), np.zeros(num_days))
        np.testing.assert_array_equal(probability_ax.lines[3].get_ydata(), np.zeros(num_days))
        assert np.isnan(probability_ax.lines[5].get_ydata()).all()
        assert not increase_ax.lines
        assert increase_ax.get_xlim() == probability_ax.get_xlim()
        for category_idx, collection in enumerate(increase_ax.collections):
            np.testing.assert_allclose(
                collection.get_facecolors()[0],
                to_rgba(f"C{category_idx}"),
            )
        assert not bar_ax.xaxis_inverted()
        assert any(f"Decision on {request_day}" in text.get_text() for text in fig.texts)
        assert all("FULL-HISTORY" not in text.get_text() for text in fig.texts)
        assert not plots.plt.fignum_exists(fig.number)
    assert artifact == original


@pytest.mark.parametrize("event_advances", [((), (), ()), ((1, 3), (2,), (3, 4)), ((62,), (), ())])
def test_bar_increase_events_use_observed_changes_and_visible_advance(
    monkeypatch: pytest.MonkeyPatch, event_advances: tuple[tuple[int, ...], ...]
) -> None:
    request_day = date(2025, 8, 1)
    num_days = 63
    before = np.ones((3, num_days), dtype=int)
    after = before.copy()
    for category_idx, days in enumerate(event_advances):
        after[category_idx, list(days)] += 1
    after[2, 5] = 0
    snapshot = {
        "start_simulation": str(request_day),
        "end_simulation": str(request_day + timedelta(days=num_days - 1)),
        "request_day": str(request_day),
        "snapshot_start_day": str(request_day),
        "rooms_by_category": [10] * 3,
        "occupancy_by_category": [[0] * num_days] * 3,
        "bar_prices_by_category": [[100] * num_days] * 3,
        "bar_levels_before_update": before.tolist(),
        "bar_levels_by_category": after.tolist(),
        "max_bar_level": 3,
        "policy_coefficients": [[0.0] * 4] * 3,
        "requests": [],
    }
    logged = []
    monkeypatch.setattr(plots.mlflow, "log_figure", lambda fig, path: logged.append((fig, path)))
    plots.log_bar_by_advance(
        {
            "checkpoint_timesteps": 800,
            "env_index": 0,
            "episode_id": 0,
            "snapshot_dates": [str(request_day)],
            "steps": [snapshot],
        }
    )
    for fig, path in logged:
        increase_ax = fig.axes[1]
        limit = num_days - 1
        for category_idx, days in enumerate(event_advances):
            np.testing.assert_array_equal(
                increase_ax.collections[category_idx].get_offsets(),
                np.column_stack((days, np.full(len(days), category_idx + 1))),
            )
        no_visible_events = not any(day <= limit for days in event_advances for day in days)
        assert bool(increase_ax.texts) == no_visible_events
        if no_visible_events:
            assert increase_ax.texts[0].get_text() == "No BAR increases on this decision day"


@pytest.mark.parametrize(
    "request_day", [date(2025, 2, 28), date(2025, 5, 15), date(2025, 8, 1), date(2025, 11, 12)]
)
def test_all_plots_exclude_other_request_days(
    monkeypatch: pytest.MonkeyPatch, request_day: date
) -> None:
    start = request_day - timedelta(days=1)
    end = request_day + timedelta(days=1)
    steps = []
    for index, day in enumerate((start, request_day, end)):
        steps.append(
            {
                "start_simulation": start.isoformat(),
                "end_simulation": end.isoformat(),
                "request_day": day.isoformat(),
                "day_index": index,
                "policy_coefficients": [[0.25 * index] * 4] * 3,
                "requests": [make_request(str(day), str(end), 1, [100 * (index + 1)])],
            }
        )
    steps[1]["requests"] = [make_request(str(request_day), str(request_day), 3, [180, 200, 220])]
    steps[1].update(
        snapshot_start_day=request_day.isoformat(),
        rooms_by_category=[10, 10, 10],
        occupancy_by_category=[[1, 2], [0, 0], [0, 0]],
        bar_prices_by_category=[[50, 200], [100, 250], [150, 300]],
        bar_levels_before_update=[[1, 1]] * 3,
        bar_levels_by_category=[[1, 2]] * 3,
        max_bar_level=3,
    )
    artifact = {
        "checkpoint_timesteps": 800,
        "env_index": 0,
        "episode_id": 0,
        "snapshot_dates": [request_day.isoformat()],
        "steps": steps,
    }
    original = deepcopy(artifact)
    logged = {}
    monkeypatch.setattr(
        plots.mlflow, "log_figure", lambda fig, path: logged.update({path.split("/")[0]: fig})
    )

    plots.log_eval_artifact_plots(artifact)

    assert len(logged) == 8
    assert "request_plots" not in logged
    np.testing.assert_array_equal(build_snapshot_metrics(steps, steps[1])["requests"], [1, 1])
    proposed = logged["proposed_price_by_advance"]
    np.testing.assert_array_equal(proposed.axes[0].lines[0].get_xdata(), [0])
    np.testing.assert_array_equal(proposed.axes[0].lines[0].get_ydata(), [200])
    for panel in proposed.axes[1:]:
        assert np.isnan(panel.lines[0].get_ydata()).all()
    for folder in ("confirmation_rate_plots", "price_acceptance_plots"):
        np.testing.assert_allclose(
            logged[folder].axes[0].collections[0].get_offsets()[:, 1], [1, 1]
        )
    expected = {
        "occupancy_plots": [100 / 30, 100 / 30],
    }
    for folder, values in expected.items():
        np.testing.assert_allclose(
            logged[folder].axes[0].lines[0].get_ydata(), values, equal_nan=True
        )
    np.testing.assert_allclose(
        logged["occupancy_plots"].axes[0].lines[1].get_ydata(), [100 * 1 / 30, 100 * 2 / 30]
    )
    np.testing.assert_array_equal(
        logged["booked_price_plots"].axes[0].lines[0].get_ydata(), [180, 200]
    )
    policy = logged["policy_diagnostics"]
    for panel in policy.axes[::3]:
        np.testing.assert_array_equal([bar.get_height() for bar in panel.containers[0]], [0.25] * 4)
        assert len(panel.patches) == 4
    np.testing.assert_array_equal(logged["bar_price_plots"].axes[0].lines[0].get_ydata(), [50, 200])
    for folder, fig in logged.items():
        for panel in fig.axes:
            is_coefficient_panel = folder == "policy_diagnostics" and any(
                panel is axis for axis in fig.axes[::3]
            )
            if is_coefficient_panel:
                assert panel.containers, folder
                assert not panel.lines, folder
                continue
            if folder == "bar_by_advance" and panel is fig.axes[1]:
                assert not panel.lines, folder
                for category_idx, collection in enumerate(panel.collections):
                    np.testing.assert_array_equal(collection.get_offsets(), [[1, category_idx + 1]])
            elif folder in {"confirmation_rate_plots", "price_acceptance_plots"}:
                assert not panel.lines, folder
                assert len(panel.collections) == 1, folder
            else:
                assert panel.lines, folder
                if folder != "proposed_price_by_advance":
                    assert not panel.collections, folder
            assert not panel.images, folder
            assert not panel.containers, folder
            assert all(line.get_linestyle() != "None" for line in panel.lines), folder
        assert any(str(request_day) in text.get_text() for text in fig.texts), folder
        assert not plots.plt.fignum_exists(fig.number)
    assert artifact == original


@pytest.mark.parametrize("num_days", [1, 3])
@pytest.mark.parametrize("with_bookings", [False, True])
def test_price_lines_deduplicate_nights_per_category(monkeypatch, num_days, with_bookings):
    start = date(2025, 1, 1)
    end = start + timedelta(days=num_days - 1)
    requests = []
    occupancy = np.zeros((3, num_days), dtype=int)
    if with_bookings:
        requests = [
            make_request(start.isoformat(), start.isoformat(), 1, [price]) for price in (0, 0, 0, 0)
        ]
        requests.extend(
            make_request(start.isoformat(), end.isoformat(), 1, [15], category=2) for _ in range(2)
        )
        requests.append(
            make_request(
                start.isoformat(),
                start.isoformat(),
                1,
                [1000],
                accepted=False,
                confirmed=False,
                category=3,
            )
        )
        occupancy[0, 0] = 4
        occupancy[1, -1] = 2
    steps = [
        {
            "start_simulation": start.isoformat(),
            "end_simulation": end.isoformat(),
            "request_day": (start + timedelta(days=index)).isoformat(),
            "day_index": index,
            "requests": requests if index == 0 else [],
            "policy_coefficients": [[0.5] * 4] * 3,
        }
        for index in range(num_days)
    ]
    grid = np.ones((3, num_days)).tolist()
    steps[0].update(
        snapshot_start_day=start.isoformat(),
        rooms_by_category=[10, 10, 10],
        occupancy_by_category=occupancy.tolist(),
        bar_prices_by_category=grid,
        bar_levels_before_update=grid,
        bar_levels_by_category=grid,
    )
    artifact = {
        "checkpoint_timesteps": 800,
        "env_index": 0,
        "episode_id": 0,
        "snapshot_dates": [start.isoformat()],
        "steps": steps,
    }
    original = deepcopy(artifact)
    logged = {}
    monkeypatch.setattr(plots.mlflow, "log_figure", lambda fig, path: logged.update({path: fig}))

    plots.log_eval_artifact_plots(artifact)

    fig = logged["booked_price_plots/2025-01-01/00000800_timesteps/env_00/episode_00000.png"]
    panels = fig.axes
    assert len(panels) == 3
    expected = [[0], [15], []] if with_bookings else [[], [], []]
    for category, panel in enumerate(panels):
        prices = np.concatenate([line.get_ydata() for line in panel.lines])
        days = np.concatenate([line.get_xdata() for line in panel.lines])
        np.testing.assert_array_equal(prices, expected[category])
        np.testing.assert_array_equal(days, [num_days if category == 1 else 1] * len(prices))
        assert len(panel.lines) == 1
        assert panel.lines[0].get_linestyle() == "-"
        assert not panel.collections
        assert panel.get_xlim() == (1, max(2, num_days))
        assert panel.get_ylim() == panels[0].get_ylim()
        assert panel.get_title() == f"Category {category + 1}"
    assert any(text.get_text() == "No confirmed bookings" for text in panels[2].texts)
    proposed = logged[
        "proposed_price_by_advance/2025-01-01/00000800_timesteps/env_00/episode_00000.png"
    ]
    if with_bookings:
        expected[2] = [1000]
    for category, panel in enumerate(proposed.axes):
        prices = np.concatenate([line.get_ydata() for line in panel.lines])
        days = np.concatenate([line.get_xdata() for line in panel.lines])
        np.testing.assert_array_equal(prices, expected[category])
        np.testing.assert_array_equal(days, [num_days - 1 if category == 1 else 0] * len(prices))
        assert len(panel.lines) == 1
        max_advance = num_days - 1 if with_bookings else 0
        assert panel.get_xlim() == (-0.5, max(1, max_advance) + 0.5)
        if with_bookings:
            assert len(panel.collections) == 1
            count = [4, 2, 1][category]
            np.testing.assert_array_equal(
                panel.collections[0].get_offsets(),
                [[num_days - 1 if category == 1 else 0, expected[category][0]]] * count,
            )
        else:
            assert not panel.collections
    assert not plots.plt.fignum_exists(fig.number)
    assert artifact == original


@pytest.mark.parametrize(
    "snapshot_day",
    [date(2025, 2, 28), date(2025, 5, 15), date(2025, 8, 1), date(2025, 11, 12)],
)
def test_proposed_price_lines_use_check_in_advance_and_full_stay_mean(
    monkeypatch: pytest.MonkeyPatch, snapshot_day: date
) -> None:
    start = date(2025, 1, 1)
    end = date(2025, 12, 31)
    requests = [
        make_request(str(snapshot_day), str(snapshot_day + timedelta(days=4)), 1, [300]),
        make_request(str(snapshot_day), str(snapshot_day), 2, [100, 120]),
        make_request(str(snapshot_day), str(snapshot_day), 2, [100, 120]),
        make_request(str(snapshot_day), str(end), 2, [400, 999], category=2),
    ]
    steps = [
        {
            "start_simulation": str(start),
            "end_simulation": str(end),
            "request_day": str(snapshot_day - timedelta(days=1)),
            "requests": [
                make_request(str(snapshot_day - timedelta(days=1)), str(snapshot_day), 1, [9999])
            ],
        },
        {"request_day": str(snapshot_day), "requests": requests},
        {"request_day": str(snapshot_day + timedelta(days=1)), "requests": []},
    ]
    artifact = {
        "steps": steps,
        "checkpoint_timesteps": 800,
        "env_index": 0,
        "episode_id": 0,
    }
    logged = []
    monkeypatch.setattr(plots.mlflow, "log_figure", lambda fig, path: logged.append(fig))

    plots.log_proposed_price_by_advance(artifact, [snapshot_day])

    assert len(logged) == 1
    panels = logged[0].axes
    assert len(panels) == 3
    assert len(panels[0].lines) == 1
    np.testing.assert_array_equal(panels[0].lines[0].get_xdata(), [0, 4])
    np.testing.assert_array_equal(panels[0].lines[0].get_ydata(), [110, 300])
    last_advance = (end - snapshot_day).days
    np.testing.assert_array_equal(panels[1].lines[0].get_xdata(), [last_advance])
    np.testing.assert_array_equal(panels[1].lines[0].get_ydata(), [699.5])
    for panel in panels:
        assert len(panel.lines) == 1
        assert panel.get_xlim() == (-0.5, last_advance + 0.5)
        assert panel.get_ylabel() == "Offered mean nightly price"
    assert panels[-1].get_xlabel() == "Advance (days from request to offered check-in)"


@pytest.mark.parametrize("confirmed_only", [False, True])
def test_conflicting_nightly_prices_are_rejected(confirmed_only: bool) -> None:
    snapshot_day = date(2025, 2, 28)
    steps = [
        {
            "request_day": str(snapshot_day),
            "requests": [
                make_request(str(snapshot_day), str(snapshot_day), 1, [100]),
                make_request(str(snapshot_day), str(snapshot_day), 1, [120]),
            ],
        }
    ]
    with pytest.raises(ValueError, match="Conflicting prices for category 1, stay day 59"):
        plots._request_price_series(
            steps,
            snapshot_day,
            date(2025, 1, 1),
            date(2025, 12, 31),
            confirmed_only=confirmed_only,
        )


def test_proposed_price_line_aggregates_offer_means_by_offered_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_day = date(2024, 2, 28)
    requests = [
        make_request("2024-02-28", "2024-03-10", 1, [80], accepted=False, confirmed=False),
        make_request("2024-02-28", "2024-02-29", 2, [50, 150]),
        make_request("2024-02-28", "2024-02-29", 1, [100]),
        make_request("2024-02-28", "2024-02-29", 3, [400, 400, 400]),
        make_request(
            "2024-02-28", "2024-02-29", 1, None, valid=False, accepted=False, confirmed=False
        ),
        make_request("2024-02-28", "2024-03-01", 2, [200, 300], category=2),
    ]
    requests[-1]["requested"] = make_request("2024-02-28", "2024-02-29", 1, None)
    requests[-1]["offer_attempts"] = [make_request("2024-02-28", "2024-02-29", 1, [999])]
    artifact = {
        "steps": [
            {
                "start_simulation": "2024-01-01",
                "end_simulation": "2024-02-28",
                "request_day": str(snapshot_day),
                "requests": requests,
            },
            {
                "request_day": "2024-02-27",
                "requests": [make_request("2024-02-27", "2024-02-29", 1, [9999])],
            },
        ],
        "checkpoint_timesteps": 800,
        "env_index": 0,
        "episode_id": 0,
    }
    original = deepcopy(artifact)
    logged = {}
    monkeypatch.setattr(plots.mlflow, "log_figure", lambda fig, path: logged.update({path: fig}))

    plots.log_proposed_price_by_advance(artifact, [snapshot_day, snapshot_day])

    assert len(logged) == 1
    fig = logged["proposed_price_by_advance/2024-02-28/00000800_timesteps/env_00/episode_00000.png"]
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), [1, 11])
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [100, 80])
    np.testing.assert_array_equal(fig.axes[1].lines[0].get_xdata(), [2])
    np.testing.assert_array_equal(fig.axes[1].lines[0].get_ydata(), [250])
    np.testing.assert_array_equal(
        fig.axes[0].collections[0].get_offsets(), [[1, 100], [1, 100], [1, 400], [11, 80]]
    )
    np.testing.assert_array_equal(fig.axes[1].collections[0].get_offsets(), [[2, 250]])
    assert any(text.get_text() == "No observed offers" for text in fig.axes[2].texts)
    assert not fig.axes[2].collections
    for panel in fig.axes[:2]:
        assert len(panel.lines) == 1
        assert panel.lines[0].get_linestyle() == "-"
        assert panel.lines[0].get_marker() == "o"
        assert len(panel.collections) == 1
        assert panel.collections[0].get_label() == "Individual offers"
        assert panel.lines[0].get_label() == "Median by advance"
        assert panel.get_legend() is not None
    assert any("Lines connect observations" in text.get_text() for text in fig.texts)
    assert not plots.plt.fignum_exists(fig.number)
    assert artifact == original


@pytest.mark.parametrize(
    ("prices", "stay"),
    [(None, 1), ([], 0), ([100], 2), ([[100]], 1), ([np.nan], 1), ([np.inf], 1)],
)
def test_proposed_prices_reject_invalid_nightly_prices(
    prices: list[float] | list[list[int]] | None, stay: int
) -> None:
    steps = [
        {
            "request_day": "2025-02-28",
            "requests": [make_request("2025-02-28", "2025-03-01", stay, prices)],
        }
    ]
    with pytest.raises(ValueError, match="one finite price per stay night"):
        _proposed_price_series(steps, date(2025, 2, 28))


@pytest.mark.parametrize(
    ("request_date", "check_in", "message"),
    [
        ("2025-02-27", "2025-03-01", "Request associated with the wrong day"),
        ("2025-02-28", "2025-02-27", "Offered check-in precedes the request date"),
    ],
)
def test_proposed_prices_reject_inconsistent_dates(
    request_date: str, check_in: str, message: str
) -> None:
    steps = [
        {
            "request_day": "2025-02-28",
            "requests": [make_request(request_date, check_in, 1, [100])],
        }
    ]
    with pytest.raises(ValueError, match=message):
        _proposed_price_series(steps, date(2025, 2, 28))


def make_request(
    request_date,
    check_in,
    days_stay,
    prices,
    *,
    valid=True,
    accepted=True,
    confirmed=True,
    category=1,
):
    return {
        "request_date": request_date,
        "check_in": check_in,
        "days_stay": days_stay,
        "category": category,
        "valid": valid,
        "price_accepted": accepted,
        "confirmed": confirmed,
        "daily_prices": prices,
    }


@pytest.fixture
def example():
    snapshot = {
        "snapshot_start_day": "2025-02-28",
        "start_simulation": "2025-01-01",
        "end_simulation": "2025-03-03",
        "rooms_by_category": [2, 2],
        "occupancy_by_category": [
            [1, 1, 0, 0],  # A
            [0, 1, 1, 0],  # B
        ],
        "bar_prices_by_category": [
            [150, 160, 170, 180],
            [250, 260, 270, 280],
        ],
    }

    steps = [
        {
            "request_day": "2025-02-20",
            "requests": [
                # A: il prezzo 80 della notte del 27 febbraio
                # NON deve entrare nel grafico dal 28 febbraio.
                make_request(
                    "2025-02-20",
                    "2025-02-27",
                    3,
                    [80, 100, 120],
                ),
            ],
        },
        {
            "request_day": "2025-02-28",
            "requests": [
                # B: arrivata esattamente al cutoff, va inclusa.
                make_request(
                    "2025-02-28",
                    "2025-03-01",
                    2,
                    [200, 300],
                    category=2,
                ),
                # C: prezzo rifiutato.
                make_request(
                    "2025-02-28",
                    "2025-03-01",
                    1,
                    [500],
                    accepted=False,
                    confirmed=False,
                ),
                # D: nessuna offerta.
                make_request(
                    "2025-02-28",
                    "2025-03-01",
                    1,
                    None,
                    valid=False,
                    accepted=False,
                    confirmed=False,
                ),
                # E: prezzo accettato, conferma finale negata.
                make_request(
                    "2025-02-28",
                    "2025-03-01",
                    1,
                    [250],
                    confirmed=False,
                ),
            ],
        },
    ]
    return steps, snapshot


def test_known_values(example):
    steps, snapshot = example
    original = deepcopy(example)

    result = build_snapshot_metrics(steps, snapshot)

    expected = {
        "day": [59, 60, 61, 62],
        "requests": [0, 4, 1, 0],
        "offers": [0, 3, 1, 0],
        "confirmed": [0, 1, 1, 0],
        "booked_revenue": [500, 0, 0, 0],
        "confirmation_rate": [np.nan, 0.25, 1, np.nan],
        "price_acceptance_rate": [np.nan, 2 / 3, 1, np.nan],
        "occupancy_pct": [0, 25, 25, 0],
        "total_occupancy_pct": [25, 50, 25, 0],
        "bar_prices": [
            [150, 160, 170, 180],
            [250, 260, 270, 280],
        ],
        "booked_price_points": [[60, 2, 200], [61, 2, 300]],
    }

    assert result.keys() == expected.keys()
    for name, values in expected.items():
        np.testing.assert_allclose(result[name], values, equal_nan=True, err_msg=name)

    # Il calcolo non deve alterare i dati originali.
    assert example == original


@pytest.mark.parametrize(
    ("original_check_in", "original_stay"), [("2025-02-28", 3), ("2025-03-03", 1)]
)
def test_alternative_offer_keeps_demand_on_original_dates(
    example: tuple[list[dict], dict], original_check_in: str, original_stay: int
) -> None:
    _, snapshot = deepcopy(example)
    snapshot["occupancy_by_category"] = [[0, 0, 0, 0], [0, 1, 1, 0]]
    original = make_request("2025-02-28", original_check_in, original_stay, None, category=1)
    offer = make_request("2025-02-28", "2025-03-01", 2, [200, 300], category=2)
    offer["requested"] = original
    steps = [{"request_day": "2025-02-28", "requests": [offer]}]
    before = deepcopy(steps)

    result = build_snapshot_metrics(steps, snapshot)

    expected_demand = [1, 1, 1, 0] if original_stay == 3 else [0, 0, 0, 1]
    expected_rate = [1, 1, 1, np.nan] if original_stay == 3 else [np.nan, np.nan, np.nan, 1]
    np.testing.assert_array_equal(result["requests"], expected_demand)
    np.testing.assert_allclose(result["confirmation_rate"], expected_rate, equal_nan=True)
    np.testing.assert_array_equal(result["confirmed"], [0, 1, 1, 0])
    np.testing.assert_array_equal(result["booked_revenue"], [500, 0, 0, 0])
    np.testing.assert_allclose(
        result["price_acceptance_rate"], [np.nan, 1, 1, np.nan], equal_nan=True
    )
    np.testing.assert_array_equal(result["booked_price_points"], [[60, 2, 200], [61, 2, 300]])
    assert steps == before


def test_requests_after_cutoff_do_not_change_any_metric(example):
    steps, snapshot = example
    baseline = build_snapshot_metrics(steps, snapshot)

    future_step = {
        "request_day": "2025-03-01",
        "requests": [
            make_request(
                "2025-03-01",
                "2025-03-02",
                1,
                [9999],
                category=2,
            ),
        ],
    }

    result = build_snapshot_metrics(steps + [future_step], snapshot)

    for name in baseline:
        np.testing.assert_allclose(result[name], baseline[name], equal_nan=True, err_msg=name)


def test_no_requests_means_zero_counts_and_undefined_ratios(example):
    _, snapshot = deepcopy(example)

    result = build_snapshot_metrics([], snapshot)

    for name in (
        "requests",
        "confirmed",
        "booked_revenue",
        "occupancy_pct",
    ):
        np.testing.assert_array_equal(result[name], [0, 0, 0, 0])

    for name in (
        "confirmation_rate",
        "price_acceptance_rate",
    ):
        assert np.isnan(result[name]).all(), name

    # Le tariffe correnti esistono anche senza prenotazioni.
    np.testing.assert_array_equal(result["bar_prices"], snapshot["bar_prices_by_category"])


def test_stay_is_clipped_but_full_revenue_belongs_to_booking_day(example):
    _, snapshot = deepcopy(example)
    snapshot["occupancy_by_category"] = [
        [0, 0, 1, 1],
        [0, 0, 0, 0],
    ]

    steps = [
        {
            "request_day": "2025-02-28",
            "requests": [
                make_request(
                    "2025-02-28",
                    "2025-03-02",
                    3,
                    [70, 80, 90],
                ),
            ],
        },
    ]

    result = build_snapshot_metrics(steps, snapshot)

    np.testing.assert_array_equal(result["booked_revenue"], [240, 0, 0, 0])
    np.testing.assert_array_equal(result["confirmed"], [0, 0, 1, 1])


def test_separate_calls_do_not_accumulate_episodes(example):
    steps, snapshot = example
    first = build_snapshot_metrics(steps, snapshot)

    empty_snapshot = deepcopy(snapshot)
    empty_snapshot["occupancy_by_category"] = [[0] * 4, [0] * 4]

    empty = build_snapshot_metrics([], empty_snapshot)
    np.testing.assert_array_equal(empty["booked_revenue"], [0, 0, 0, 0])

    repeated = build_snapshot_metrics(steps, snapshot)
    for name in first:
        np.testing.assert_allclose(repeated[name], first[name], equal_nan=True, err_msg=name)


def test_requests_before_snapshot_do_not_change_any_metric(example):
    steps, snapshot = deepcopy(example)
    result = build_snapshot_metrics(steps, snapshot)
    daily = build_snapshot_metrics(steps[1:], snapshot)

    for name in result:
        np.testing.assert_allclose(result[name], daily[name], equal_nan=True, err_msg=name)


def test_missing_nightly_price_is_rejected(example):
    steps, snapshot = deepcopy(example)
    steps[1]["requests"][0]["daily_prices"] = [200]

    with pytest.raises(ValueError, match="Daily prices"):
        build_snapshot_metrics(steps, snapshot)
