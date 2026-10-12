"""Diagnostica (Passo 0): curva di accettazione del modello logistico per livello BAR.

Non usa l'ambiente né la policy: dato il modello di accettazione e la BAR table,
calcola per ogni categoria e stagione
- la probabilita' di accettazione a ogni livello BAR (reference price deterministico,
  funzione di categoria, stagione e anticipo);
- il ricavo atteso per richiesta (prezzo * probabilita') e il livello che lo massimizza.

Opzionalmente confronta piu' scale del reference price (prezzo base e scala logistica moltiplicati insieme).

Uso (dalla root del progetto):
    uv run python -m pricing_advertising.diagnostics.acceptance_curve
    uv run python -m pricing_advertising.diagnostics.acceptance_curve --scales 1.0 0.85 0.7
    uv run python -m pricing_advertising.diagnostics.acceptance_curve --advance 90
"""

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from pricing_advertising.acceptance.logistic_function import LogisticAcceptanceModel
from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.pricing_model import bar_table
from pricing_advertising.pricing_model.bar_table import set_bar_table_from_csv
from pricing_advertising.utils.constants import (
    BOOKING_WINDOW,
    HIGH_SEASON,
    LOGISTIC_ACCEPTANCE_ADVANCE_MAX,
    LOGISTIC_ACCEPTANCE_ADVANCE_MIN,
    LOGISTIC_ACCEPTANCE_AVG_PRICES,
    LOGISTIC_ACCEPTANCE_BETA,
    LOGISTIC_ACCEPTANCE_HIGH_SEASON_MULTIPLIERS,
    LOGISTIC_ACCEPTANCE_STDS,
)

BAR_CSV = Path("data/bar_table.csv")
OUTPUT_DIR = Path("tmp/acceptance_diagnostics")
# Date di esempio: aprile e' bassa stagione, luglio alta (vedi HIGH_SEASON)
SEASON_DATES = {"low": date(2025, 4, 15), "high": date(2025, 7, 15)}
CATEGORIES = (1, 2, 3)


def build_model(
    scale: float, high_season_multiplier: float | None, beta: float
) -> LogisticAcceptanceModel:
    multipliers = dict(LOGISTIC_ACCEPTANCE_HIGH_SEASON_MULTIPLIERS)
    if high_season_multiplier is not None:
        multipliers = dict.fromkeys(multipliers, high_season_multiplier)
    return LogisticAcceptanceModel(
        beta=beta,
        avg_prices={c: p * scale for c, p in LOGISTIC_ACCEPTANCE_AVG_PRICES.items()},
        st_deviations={c: s * scale for c, s in LOGISTIC_ACCEPTANCE_STDS.items()},
        high_season_multipliers=multipliers,
        advance_max=LOGISTIC_ACCEPTANCE_ADVANCE_MAX,
        advance_min=LOGISTIC_ACCEPTANCE_ADVANCE_MIN,
        booking_window=BOOKING_WINDOW,
    )


def reference_price(
    model: LogisticAcceptanceModel, category: int, season: str, advance: int
) -> float:
    check_in = SEASON_DATES[season]
    return model.compute_reference_price(
        RoomCategory(category),
        t_request=check_in - timedelta(days=advance),
        t_reservation=check_in,
        days_stay=1,
    )


def acceptance_by_level(
    model: LogisticAcceptanceModel,
    category: int,
    season: str,
    prices: np.ndarray,
    advance: int,
) -> np.ndarray:
    """Probabilita' di accettazione di ogni prezzo rispetto al reference price deterministico."""
    return model.logistic_function(
        proposed_price=prices,
        reference_price=reference_price(model, category, season, advance),
        beta=model.beta,
        standard_deviation=model.st_deviations[category],
    )


def first_level_below(probs: np.ndarray, threshold: float) -> int | None:
    below = np.flatnonzero(probs < threshold)
    return int(below[0]) if below.size else None


def run(
    scales: list[float], high_season_multiplier: float | None, beta: float, advance: int
) -> None:
    set_bar_table_from_csv(BAR_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assert SEASON_DATES["low"].month not in HIGH_SEASON
    assert SEASON_DATES["high"].month in HIGH_SEASON

    rows: list[dict] = []
    curves: dict[tuple[float, str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for scale in scales:
        model = build_model(scale, high_season_multiplier, beta)
        for season in SEASON_DATES:
            for category in CATEGORIES:
                bar = np.array(bar_table.BAR_TABLE[RoomCategory(category)], dtype=float)
                levels = np.arange(1, len(bar))  # il livello 0 ha prezzo 0
                prices = bar[1:]
                probs = acceptance_by_level(model, category, season, prices, advance)
                revenue = prices * probs
                best = int(np.argmax(revenue))
                curves[(scale, season, category)] = (levels, probs, revenue)
                rows.append(
                    {
                        "scale": scale,
                        "season": season,
                        "category": category,
                        "beta": beta,
                        "advance": advance,
                        "ref_price": round(reference_price(model, category, season, advance), 1),
                        "price_scale": round(model.st_deviations[category], 1),
                        "n_levels": len(levels),
                        "max_bar_price": float(prices[-1]),
                        "level_p_below_95": first_level_below(probs, 0.95),
                        "level_p_below_50": first_level_below(probs, 0.50),
                        "level_p_below_10": first_level_below(probs, 0.10),
                        "p_at_top_level": round(float(probs[-1]), 4),
                        "best_revenue_level": int(levels[best]),
                        "best_revenue_price": float(prices[best]),
                        "p_at_best_level": round(float(probs[best]), 4),
                    }
                )

    csv_path = OUTPUT_DIR / "acceptance_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Riepilogo salvato in {csv_path}")
    for r in rows:
        logger.info(
            f"scale={r['scale']} {r['season']:>4} cat{r['category']} anticipo={r['advance']}: "
            f"ref={r['ref_price']} (scala {r['price_scale']}) | p<95% dal livello {r['level_p_below_95']}, "
            f"p<50% dal {r['level_p_below_50']}, p<10% dal {r['level_p_below_10']} "
            f"(su {r['n_levels']}) | ottimo ricavo: livello {r['best_revenue_level']} "
            f"(prezzo {r['best_revenue_price']:.0f}, p={r['p_at_best_level']:.2f}) | "
            f"p al livello massimo={r['p_at_top_level']:.3f}"
        )

    plot_curves(
        curves, scales, metric=1, ylabel="P(accettazione)", filename="acceptance_probability.png"
    )
    plot_curves(
        curves,
        scales,
        metric=2,
        ylabel="Ricavo atteso per richiesta",
        filename="expected_revenue.png",
    )


def plot_curves(
    curves: dict[tuple[float, str, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    scales: list[float],
    *,
    metric: int,
    ylabel: str,
    filename: str,
) -> None:
    """metric: 1 = probabilita' di accettazione, 2 = ricavo atteso (con marcatore sul massimo)."""
    fig, axes = plt.subplots(
        len(SEASON_DATES), len(CATEGORIES), figsize=(15, 8), sharex="col", squeeze=False
    )
    for i, season in enumerate(SEASON_DATES):
        for j, category in enumerate(CATEGORIES):
            ax = axes[i][j]
            for scale in scales:
                levels, probs, revenue = curves[(scale, season, category)]
                y = probs if metric == 1 else revenue
                (line,) = ax.plot(levels, y, label=f"scale={scale}")
                if metric == 2:
                    k = int(np.argmax(y))
                    ax.plot(levels[k], y[k], "o", color=line.get_color())
            ax.set_title(f"Categoria {category} - {season} season")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(SEASON_DATES) - 1:
                ax.set_xlabel("Livello BAR")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=130)
    plt.close(fig)
    logger.info(f"Grafico salvato in {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="Fattori che moltiplicano prezzo base e scala logistica (1.0 = valori attuali).",
    )
    parser.add_argument(
        "--high-season-multiplier",
        type=float,
        default=None,
        help="Sostituisce il moltiplicatore di alta stagione dell'accettazione (default: costanti).",
    )
    parser.add_argument("--beta", type=float, default=LOGISTIC_ACCEPTANCE_BETA)
    parser.add_argument(
        "--advance",
        type=int,
        default=0,
        help="Giorni di anticipo tra richiesta e check-in (0 = last-minute).",
    )
    args = parser.parse_args()
    run(args.scales, args.high_season_multiplier, args.beta, args.advance)
