from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Denominatore zero = dato non definito, non tasso/prezzo zero."""
    result = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def build_snapshot_metrics(
    steps: list[dict[str, Any]],
    snapshot: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Costruisce una fotografia usando solo richieste note entro il cutoff."""
    cutoff = date.fromisoformat(snapshot["snapshot_start_day"])
    start = date.fromisoformat(snapshot["start_simulation"])
    end = date.fromisoformat(snapshot["end_simulation"])

    if not start <= cutoff <= end:
        raise ValueError("Data della fotografia fuori dalla simulazione.")

    size = (end - cutoff).days + 1
    first_day = (cutoff - start).days + 1

    requests = np.zeros(size, dtype=np.int64)
    offers = np.zeros(size, dtype=np.int64)
    price_accepted = np.zeros(size, dtype=np.int64)
    confirmed = np.zeros(size, dtype=np.int64)
    revenue = np.zeros(size, dtype=float)

    for step in steps:
        request_day = date.fromisoformat(step["request_day"])

        # Esclude esplicitamente qualsiasi informazione arrivata dopo cutoff.
        if request_day > cutoff:
            continue

        for request in step["requests"]:
            if date.fromisoformat(request["request_date"]) != request_day:
                raise ValueError("Richiesta associata al giorno sbagliato.")

            is_valid = bool(request["valid"])
            is_accepted = bool(request["price_accepted"])
            is_confirmed = bool(request["confirmed"])

            if is_accepted and not is_valid:
                raise ValueError("Prezzo accettato senza offerta.")
            if is_confirmed and not is_accepted:
                raise ValueError("Conferma senza accettazione del prezzo.")

            stay = int(request["days_stay"])
            check_in = date.fromisoformat(request["check_in"])
            prices = request["daily_prices"]

            if is_valid:
                if prices is None or len(prices) != stay:
                    raise ValueError("Prezzi giornalieri non allineati al soggiorno.")

            # Intersezione tra soggiorno e intervallo [cutoff, end].
            # Gestisce anche soggiorni iniziati PRIMA del cutoff.
            offset = (check_in - cutoff).days
            first_night = max(0, -offset)
            stop_night = min(stay, size - offset)

            for night in range(first_night, stop_night):
                idx = offset + night

                requests[idx] += 1
                offers[idx] += int(is_valid)
                price_accepted[idx] += int(is_accepted)
                confirmed[idx] += int(is_confirmed)

                if is_confirmed:
                    revenue[idx] += float(prices[night])

    occupancy = np.asarray(snapshot["occupancy_by_category"], dtype=float)
    bar_prices = np.asarray(snapshot["bar_prices_by_category"], dtype=float)
    rooms = np.asarray(snapshot["rooms_by_category"], dtype=float)

    expected_shape = (len(rooms), size)
    if occupancy.shape != expected_shape or bar_prices.shape != expected_shape:
        raise ValueError("Dimensioni della fotografia non coerenti con le date.")

    if rooms.sum() <= 0:
        raise ValueError("Capacità totale dell'hotel non positiva.")

    # Nel modello attuale ogni richiesta confermata occupa una camera
    # per ciascuna notte del soggiorno.
    if not np.allclose(occupancy.sum(axis=0), confirmed):
        raise ValueError(
            "Le conferme ricostruite non coincidono con la fotografia dell'occupazione."
        )

    return {
        "day": np.arange(first_day, first_day + size),
        "requests": requests,
        "confirmed": confirmed,
        "booked_revenue": revenue,
        "booked_mean_price": _ratio(revenue, confirmed),
        "confirmation_rate": _ratio(confirmed, requests),
        "price_acceptance_rate": _ratio(price_accepted, offers),
        "occupancy_pct": occupancy.sum(axis=0) / rooms.sum() * 100.0,
        "bar_prices": bar_prices,
    }


def log_eval_artifact_plots(artifact: dict[str, Any]) -> None:
    """Genera PNG separati per data, checkpoint, ambiente ed episodio."""
    steps = artifact["steps"]
    if not steps:
        raise ValueError("Episodio di evaluation vuoto.")

    start = date.fromisoformat(steps[0]["start_simulation"])
    end = date.fromisoformat(steps[0]["end_simulation"])

    # Controllo di continuità: nessun giorno mancante o duplicato.
    for index, step in enumerate(steps):
        expected_day = start + timedelta(days=index)
        if step["request_day"] != expected_day.isoformat() or int(step["day_index"]) != index:
            raise ValueError("Sequenza dei giorni dell'episodio non continua.")

    # end_simulation è incluso tra i giorni di decisione.
    expected_steps = (end - start).days + 1

    if len(steps) != expected_steps or steps[-1]["request_day"] != end.isoformat():
        raise ValueError("Ultimo step inatteso o episodio incompleto.")

    snapshots = [step for step in steps if "snapshot_start_day" in step]
    expected_dates = set(artifact["snapshot_dates"])
    actual_dates = {step["snapshot_start_day"] for step in snapshots}

    if actual_dates != expected_dates:
        raise ValueError(
            f"Fotografie mancanti o inattese: "
            f"attese={sorted(expected_dates)}, trovate={sorted(actual_dates)}"
        )

    checkpoint = int(artifact["checkpoint_timesteps"])
    env_idx = int(artifact["env_index"])
    episode_id = int(artifact["episode_id"])

    # Chiave, cartella, etichetta Y, limiti Y.
    specifications = [
        (
            "booked_mean_price",
            "booked_price_plots",
            "Mean booked nightly price",
            None,
        ),
        (
            "booked_revenue",
            "booked_revenue_plots",
            "Booked revenue per night",
            None,
        ),
        (
            "requests",
            "request_plots",
            "Requests covering this night",
            None,
        ),
        (
            "confirmation_rate",
            "confirmation_rate_plots",
            "Confirmed / all requests",
            (0, 1),
        ),
        (
            "price_acceptance_rate",
            "price_acceptance_plots",
            "Price accepted / offers",
            (0, 1),
        ),
        (
            "occupancy_pct",
            "occupancy_plots",
            "Booked occupancy (%)",
            (0, 100),
        ),
        (
            "bar_prices",
            "bar_price_plots",
            "Current BAR nightly price",
            None,
        ),
    ]

    for snapshot in snapshots:
        cutoff = snapshot["snapshot_start_day"]
        metrics = build_snapshot_metrics(steps, snapshot)
        x = metrics["day"]

        for key, folder, ylabel, ylim in specifications:
            fig, ax = plt.subplots(figsize=(12, 5))

            try:
                if key == "bar_prices":
                    for category, prices in enumerate(metrics[key], start=1):
                        ax.plot(x, prices, label=f"Category {category}")
                    ax.legend()
                else:
                    y = metrics[key]
                    ax.plot(x, y, marker=".", markersize=2)

                    if not np.isfinite(y).any():
                        ax.text(
                            0.5,
                            0.5,
                            "No observations",
                            transform=ax.transAxes,
                            ha="center",
                        )

                ax.set_xlabel("Stay day (1 = simulation start)")
                ax.set_ylabel(ylabel)
                ax.set_title(
                    f"{ylabel} — as of {cutoff}\n"
                    f"Training timesteps: {checkpoint} | "
                    f"Env: {env_idx} | Episode: {episode_id}"
                )
                ax.set_xlim(float(x[0]) - 0.5, float(x[-1]) + 0.5)

                if ylim is not None:
                    ax.set_ylim(*ylim)

                ax.grid(True, alpha=0.3)
                fig.tight_layout()

                mlflow.log_figure(
                    fig,
                    (
                        f"{folder}/{cutoff}/"
                        f"{checkpoint:08d}_timesteps/"
                        f"env_{env_idx:02d}/episode_{episode_id:05d}.png"
                    ),
                )
            finally:
                plt.close(fig)
