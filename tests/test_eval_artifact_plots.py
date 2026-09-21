from copy import deepcopy

import numpy as np
import pytest

from pricing_advertising.plots.eval_artifact_plots import build_snapshot_metrics


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
        "requests": [1, 5, 1, 0],
        "confirmed": [1, 2, 1, 0],
        "booked_revenue": [100, 320, 300, 0],
        "booked_mean_price": [100, 160, 300, np.nan],
        "confirmation_rate": [1, 0.4, 1, np.nan],
        "price_acceptance_rate": [1, 0.75, 1, np.nan],
        "occupancy_pct": [25, 50, 25, 0],
        "bar_prices": [
            [150, 160, 170, 180],
            [250, 260, 270, 280],
        ],
    }

    assert result.keys() == expected.keys()
    for name, values in expected.items():
        np.testing.assert_allclose(result[name], values, equal_nan=True, err_msg=name)

    # Il calcolo non deve alterare i dati originali.
    assert example == original


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
    snapshot["occupancy_by_category"] = [[0] * 4, [0] * 4]

    result = build_snapshot_metrics([], snapshot)

    for name in (
        "requests",
        "confirmed",
        "booked_revenue",
        "occupancy_pct",
    ):
        np.testing.assert_array_equal(result[name], [0, 0, 0, 0])

    for name in (
        "booked_mean_price",
        "confirmation_rate",
        "price_acceptance_rate",
    ):
        assert np.isnan(result[name]).all(), name

    # Le tariffe correnti esistono anche senza prenotazioni.
    np.testing.assert_array_equal(result["bar_prices"], snapshot["bar_prices_by_category"])


def test_stay_is_clipped_at_plot_end_without_redistributing_revenue(example):
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

    # La notte del 4 marzo è fuori dal grafico:
    # il suo ricavo 90 non va né mostrato né redistribuito.
    np.testing.assert_array_equal(result["booked_revenue"], [0, 0, 70, 80])
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


def test_inconsistent_occupancy_is_rejected(example):
    steps, snapshot = deepcopy(example)
    snapshot["occupancy_by_category"][0][0] = 0

    with pytest.raises(ValueError, match="conferme ricostruite"):
        build_snapshot_metrics(steps, snapshot)


def test_missing_nightly_price_is_rejected(example):
    steps, snapshot = deepcopy(example)
    steps[0]["requests"][0]["daily_prices"] = [80, 100]

    with pytest.raises(ValueError, match="Prezzi giornalieri"):
        build_snapshot_metrics(steps, snapshot)
