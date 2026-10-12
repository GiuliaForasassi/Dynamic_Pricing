"""Diagnostica: sensibilita' dell'accettazione al moltiplicatore di alta stagione.

Per ogni categoria sovrappone, nella stessa figura, la curva di accettazione e il ricavo atteso
in alta stagione ottenuti con diversi moltiplicatori. Il moltiplicatore 1.0 coincide con la
bassa stagione, che non dipende dal moltiplicatore.

Uso (dalla root del progetto):
    uv run python -m pricing_advertising.diagnostics.acceptance_multiplier_sensitivity
    uv run python -m pricing_advertising.diagnostics.acceptance_multiplier_sensitivity \
        --multipliers 1.0 1.1 1.2 1.3 --advance 0
"""

import argparse

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from pricing_advertising.diagnostics.acceptance_curve import (
    BAR_CSV,
    CATEGORIES,
    OUTPUT_DIR,
    acceptance_by_level,
    build_model,
)
from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.pricing_model import bar_table
from pricing_advertising.pricing_model.bar_table import set_bar_table_from_csv
from pricing_advertising.utils.constants import LOGISTIC_ACCEPTANCE_BETA

FILENAME = "multiplier_sensitivity.png"


def run(multipliers: list[float], beta: float, advance: int) -> None:
    set_bar_table_from_csv(BAR_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, len(CATEGORIES), figsize=(15, 8), sharex="col", squeeze=False)
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(multipliers)))
    for j, category in enumerate(CATEGORIES):
        bar = np.array(bar_table.BAR_TABLE[RoomCategory(category)], dtype=float)
        levels = np.arange(1, len(bar))  # il livello 0 ha prezzo 0
        prices = bar[1:]
        for multiplier, color in zip(multipliers, colors, strict=True):
            model = build_model(1.0, multiplier, beta)
            probs = acceptance_by_level(model, category, "high", prices, advance)
            revenue = prices * probs
            best = int(np.argmax(revenue))
            label = f"x{multiplier:g}"
            axes[0][j].plot(levels, probs, color=color, label=label)
            (line,) = axes[1][j].plot(levels, revenue, color=color, label=label)
            axes[1][j].plot(levels[best], revenue[best], "o", color=line.get_color())
            logger.info(
                f"mult={multiplier:g} cat{category}: ottimo livello {levels[best]} "
                f"(prezzo {prices[best]:.0f}, p={probs[best]:.2f}) | p al livello massimo={probs[-1]:.3f}"
            )
        axes[0][j].set_title(f"Categoria {category} - high season")
        for ax in axes[:, j]:
            ax.grid(alpha=0.3)
        axes[1][j].set_xlabel("Livello BAR")
    axes[0][0].set_ylabel("P(accettazione)")
    axes[1][0].set_ylabel("Ricavo atteso per richiesta")
    axes[0][0].legend(title="Moltiplicatore")
    fig.suptitle(f"Sensibilita' al moltiplicatore di alta stagione (anticipo={advance} giorni)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / FILENAME, dpi=130)
    plt.close(fig)
    logger.info(f"Grafico salvato in {OUTPUT_DIR / FILENAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--multipliers", type=float, nargs="+", default=[1.0, 1.1, 1.2, 1.3])
    parser.add_argument("--beta", type=float, default=LOGISTIC_ACCEPTANCE_BETA)
    parser.add_argument("--advance", type=int, default=0)
    args = parser.parse_args()
    run(args.multipliers, args.beta, args.advance)
