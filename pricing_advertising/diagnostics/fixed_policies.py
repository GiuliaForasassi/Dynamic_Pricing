"""Diagnostica: confronta policy fisse e PPO sugli stessi seed.

Misura effetto delle azioni, saturazione BAR, domanda e scala dei ricavi.
I confronti tra seed sono appaiati, ma non isolano tutte le fonti di rumore.

Uso (dalla root del progetto):
    uv run python -m pricing_advertising.diagnostics.fixed_policies
"""

import csv
import sys
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch
from loguru import logger
from stable_baselines3 import PPO

from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.pricing_model.bar_table import set_bar_table_from_csv
from pricing_advertising.pricing_model.bernstein_model import BernsteinProbModel
from pricing_advertising.utils.constants import NUM_COEFFICIENTS, NUM_WINDOWS, SEEDS, WINDOW_SIZE

# ============ CONFIGURAZIONE (uguale a __main__.py) ============
START = date(2025, 1, 1)
END = date(2025, 12, 31)
ROOMS = (100, 100, 100)
AVG_NUM_REQUESTS = 250
N_SEEDS = 20
TEST_SEED_START = SEEDS[0] + 30_000
MODEL_PATH = Path("models/reward_scaled/b514909fbd3240b485a8b1dc26a9b590/final_model.zip")
RESULTS_PATH = Path("tmp/reward_scaled_500k_holdout/policy_comparison.csv")
# Giorni dall'inizio in cui fotografare i livelli BAR delle date future
SNAPSHOT_DAYS = (30, 90, 180, 270)
LOOKAHEAD = 90

Policy = Callable[[dict], np.ndarray]


def make_policies(n_categories: int, seed: int) -> dict[str, Policy]:
    shape = (n_categories, 2, NUM_COEFFICIENTS)
    rng = np.random.default_rng(seed)
    return {
        "theta=0 (sale piano)": lambda _obs: np.zeros(shape, np.float32),
        "theta=0.5": lambda _obs: np.full(shape, 0.5, np.float32),
        "theta=1 (sale veloce)": lambda _obs: np.ones(shape, np.float32),
        "uniforme casuale": lambda _obs: rng.uniform(0, 1, shape).astype(np.float32),
    }


def raw_demand_windows(env: PricingEnv) -> np.ndarray:
    current_day = (env.today - env.start_simulation).days
    values = np.zeros((env.num_room_categories, NUM_WINDOWS), dtype=float)
    for window in range(NUM_WINDOWS):
        start = current_day + 1 + window * WINDOW_SIZE
        stop = min(start + WINDOW_SIZE, env.extended_simulation_days)
        width = max(1, stop - start)
        values[:, window] = env.demand[:, start:stop].sum(axis=1) / max(
            1, width * env.avg_num_requests
        )
    return values


def run_episode(policy: Policy, seed: int, gamma: float) -> dict:
    hotel = Hotel(
        num_category1_rooms=ROOMS[0],
        num_category2_rooms=ROOMS[1],
        num_category3_rooms=ROOMS[2],
    )
    env = PricingEnv(
        start_simulation=START,
        end_simulation=END,
        hotel=hotel,
        avg_num_requests=AVG_NUM_REQUESTS,
        seed=seed,
    )
    torch.manual_seed(seed)

    obs, _ = env.reset(seed=seed)
    max_level = env.n_levels - 1

    revenue, offers, confirmed = 0.0, 0, 0
    nightly_by_season: dict[int, list[float]] = {0: [], 1: []}
    bar_snapshots: dict[int, tuple[float, float]] = {}

    discounted_revenue = 0.0
    terminated = False
    daily_diagnostics = []
    demand_before_action = []
    while not terminated:
        day_index = (env.today - env.start_simulation).days
        levels_before = env.bar_levels.copy()
        demand_before_action.append(raw_demand_windows(env))
        action = policy(obs)
        obs, reward, terminated, _, info = env.step(action)

        before_offered, after_offered = [], []
        for offer in env.last_valid_requests:
            category_index = offer.category.value - 1
            check_in = (offer.t_reservation - env.start_simulation).days
            stop = check_in + offer.days_stay
            start = max(check_in, day_index + 1)
            before_offered.extend(levels_before[category_index, start:stop])
            after_offered.extend(env.bar_levels[category_index, start:stop])

        before_offered = np.asarray(before_offered)
        after_offered = np.asarray(after_offered)
        nightly_prices = info["step_metrics"]["confirmed_nightly_prices"]
        has_offers = before_offered.size > 0
        daily_diagnostics.append(
            {
                "day_index": day_index,
                "reward": float(reward),
                "offered_future_nights": int(before_offered.size),
                "offered_at_max_before": int(np.count_nonzero(before_offered == max_level)),
                "offered_changed": int(np.count_nonzero(after_offered > before_offered)),
                "mean_offered_level": float(after_offered.mean()) if has_offers else np.nan,
                "sold_nights": int(nightly_prices.size),
                "sold_at_zero": int(np.count_nonzero(nightly_prices == 0)),
                "arrival_level_mean": float(levels_before[:, day_index].mean()),
                "arrival_at_max_fraction": float((levels_before[:, day_index] == max_level).mean()),
            }
        )
        metrics = info["step_metrics"]

        revenue += float(reward)
        discounted_revenue += gamma ** int(metrics["day_index"]) * float(reward)
        offers += metrics["num_offers"]
        confirmed += metrics["num_confirmed"]

        # Prezzi per notte confermati, divisi per stagione della notte
        prices = iter(metrics["confirmed_nightly_prices"])
        for request in metrics["confirmed_requests"]:
            for night in range(request.days_stay):
                season = env.get_seasonality(request.t_reservation + timedelta(days=night))
                nightly_by_season[season].append(float(next(prices)))

        # Livello BAR medio e quota al massimo sulle prossime LOOKAHEAD date
        day = metrics["day_index"]
        if day in SNAPSHOT_DAYS:
            future = env.bar_levels[:, day + 1 : day + 1 + LOOKAHEAD]
            bar_snapshots[day] = (float(future.mean()), float((future == max_level).mean()))

    occupancy = env.occupancy[:, : env.total_days].sum(axis=1) / (np.array(ROOMS) * env.total_days)
    env.close()

    return {
        "seed": seed,
        "revenue": revenue,
        "conversion": confirmed / max(1, offers),
        "price_low": np.mean(nightly_by_season[0]) if nightly_by_season[0] else float("nan"),
        "price_high": np.mean(nightly_by_season[1]) if nightly_by_season[1] else float("nan"),
        "occupancy": occupancy,
        "bar_snapshots": bar_snapshots,
        "discounted_revenue": discounted_revenue,
        "daily_diagnostics": daily_diagnostics,
        "demand_before_action": np.asarray(demand_before_action),
    }


def summarize_diagnostics(result: dict) -> dict[str, float]:
    daily = result["daily_diagnostics"]
    offered_nights = sum(row["offered_future_nights"] for row in daily)
    sold_nights = sum(row["sold_nights"] for row in daily)
    rewards = np.asarray([row["reward"] for row in daily])
    demand = result["demand_before_action"]
    summary = {
        "offered_at_max_before_fraction": (
            sum(row["offered_at_max_before"] for row in daily) / offered_nights
            if offered_nights
            else float("nan")
        ),
        "offered_changed_fraction": (
            sum(row["offered_changed"] for row in daily) / offered_nights
            if offered_nights
            else float("nan")
        ),
        "sold_at_zero_fraction": (
            sum(row["sold_at_zero"] for row in daily) / sold_nights if sold_nights else float("nan")
        ),
        "arrival_at_max_fraction": float(
            np.mean([row["arrival_at_max_fraction"] for row in daily])
        ),
        "demand_clipped_fraction": float((demand > 1).mean()),
        "reward_zero_fraction": float((rewards == 0).mean()),
        "reward_std_eur": float(rewards.std()),
    }
    for label, quantile in (("p05", 0.05), ("p50", 0.5), ("p95", 0.95)):
        reward_value = float(np.quantile(rewards, quantile))
        summary[f"reward_{label}_eur"] = reward_value
        summary[f"demand_{label}"] = float(np.quantile(demand, quantile))
    return summary


def write_episode_diagnostics(result: dict, output_dir: Path, name: str) -> None:
    suffix = f"{name.replace(' ', '_')}_{result['seed']}.csv"
    daily = result["daily_diagnostics"]
    with (output_dir / f"daily_{suffix}").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(daily[0]))
        writer.writeheader()
        writer.writerows(daily)
    with (output_dir / f"demand_{suffix}").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["day_index", "category", "window_index", "raw_demand", "clipped"])
        for day_index, values in enumerate(result["demand_before_action"]):
            for (category, window), value in np.ndenumerate(values):
                writer.writerow([day_index, category + 1, window, value, bool(value > 1)])


def paired_differences(runs: list[dict], baseline: list[dict], metric: str) -> np.ndarray:
    reference = {run["seed"]: run[metric] for run in baseline}
    if (
        len(reference) != len(baseline)
        or len({run["seed"] for run in runs}) != len(runs)
        or {run["seed"] for run in runs} != set(reference)
    ):
        raise ValueError("Il confronto richiede gli stessi seed, senza duplicati.")
    return np.asarray([run[metric] - reference[run["seed"]] for run in runs], dtype=float)


def fmt(values: list[float], scale: float = 1.0, digits: int = 1) -> str:
    arr = np.asarray(values, dtype=float) / scale
    return f"{arr.mean():.{digits}f} ± {arr.std():.{digits}f}"


def print_bernstein_probabilities() -> None:
    occupancy_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    print("Probabilita di incremento per occupazione [0, 0.25, 0.5, 0.75, 1]:")
    for coefficient in (0.0, 0.5, 1.0):
        probabilities = BernsteinProbModel(
            np.full(NUM_COEFFICIENTS, coefficient)
        ).predict_increment_probs(occupancy_grid)
        print(f"  theta={coefficient}: {np.array2string(probabilities, precision=4)}")


def main() -> None:
    if N_SEEDS < 1:
        raise ValueError("N_SEEDS deve essere positivo.")
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    set_bar_table_from_csv(Path("data") / "bar_table.csv")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Modello non trovato: {MODEL_PATH}")

    model = PPO.load(MODEL_PATH, device="cpu")
    print(f"Modello: {MODEL_PATH.resolve()}")
    print(f"Timestep del checkpoint: {model.num_timesteps}")
    print(f"Gamma: {model.gamma}")

    model.policy.set_training_mode(False)
    if not isinstance(model.action_space, gym.spaces.Box):
        raise ValueError("Questa diagnostica richiede uno spazio di azioni Box.")
    action_space = model.action_space
    expected_shape = (len(ROOMS), 2, NUM_COEFFICIENTS)
    if action_space.shape != expected_shape:
        raise ValueError(
            f"Checkpoint action shape {action_space.shape} is incompatible with "
            f"seasonal actions {expected_shape}. Train a new PPO model."
        )
    if model.policy.squash_output:
        raise ValueError("Questa diagnostica richiede azioni non squashed.")
    action_records = []

    def ppo_policy(observation: dict, deterministic: bool) -> np.ndarray:
        observation_tensor, _ = model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = model.policy.get_distribution(observation_tensor)
            raw = distribution.get_actions(deterministic=deterministic)
            assert distribution.distribution is not None
            mean = distribution.distribution.mean
            std = distribution.distribution.stddev
        shape = action_space.shape
        raw = raw.cpu().numpy().reshape(shape)
        mean = mean.cpu().numpy().reshape(shape)
        std = std.cpu().numpy().reshape(shape)
        applied = np.clip(raw, action_space.low, action_space.high)
        action_records.append(np.stack([mean, std, raw, applied]))
        return applied.astype(np.float32)

    seeds = [TEST_SEED_START + index for index in range(N_SEEDS)]
    results: dict[str, list[dict]] = {}

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    print_bernstein_probabilities()
    print("Reward diagnostica in EUR; la normalizzazione adattiva riguarda solo il training.")
    print("Seed condivisi: le estrazioni di accettazione non sono allineate per richiesta.")
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = None

        for seed in seeds:
            policies = make_policies(n_categories=len(ROOMS), seed=seed)
            policies["PPO deterministic"] = lambda obs: ppo_policy(obs, True)
            policies["PPO stochastic"] = lambda obs: ppo_policy(obs, False)

            for name, policy in policies.items():
                t0 = perf_counter()
                action_records.clear()
                result = run_episode(policy, seed, gamma=model.gamma)
                result["diagnostic_summary"] = summarize_diagnostics(result)
                write_episode_diagnostics(result, RESULTS_PATH.parent, name)
                if action_records:
                    records = np.asarray(action_records)
                    np.savez_compressed(
                        RESULTS_PATH.parent / f"actions_{name.replace(' ', '_')}_{seed}.npz",
                        mean=records[:, 0],
                        std=records[:, 1],
                        raw=records[:, 2],
                        applied=records[:, 3],
                    )
                    print(
                        f"[{name}] medie negative: {(records[:, 0] < 0).mean():.1%}; "
                        f"azioni a zero: {(records[:, 3] == 0).mean():.1%}; "
                        f"raw < 0: {(records[:, 2] < 0).mean():.1%}; "
                        f"raw > 1: {(records[:, 2] > 1).mean():.1%}"
                    )
                results.setdefault(name, []).append(result)

                row = {
                    "policy": name,
                    "seed": seed,
                    "revenue": result["revenue"],
                    "conversion": result["conversion"],
                    "discounted_revenue": result["discounted_revenue"],
                    **result["diagnostic_summary"],
                }
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()

                print(
                    f"[{name}] seed={seed}: "
                    f"{result['revenue'] / 1e6:.3f} M€, "
                    f"{perf_counter() - t0:.1f}s"
                )

    print_results(results, RESULTS_PATH.parent)


def print_results(results: dict[str, list[dict]], output_dir: Path) -> None:
    print("\n" + "=" * 90)
    print(f"Risultati su {N_SEEDS} seed (media ± std)")
    print("=" * 90)
    for name, runs in results.items():
        print(f"\n### {name}")
        print(f"  Ricavo annuo (M€):        {fmt([r['revenue'] for r in runs], 1e6, 3)}")
        print(f"  Ricavo scontato (M€): {fmt([r['discounted_revenue'] for r in runs], 1e6, 3)}")
        print(f"  Conversione offerte:      {fmt([r['conversion'] * 100 for r in runs])} %")
        print(f"  Prezzo/notte bassa stag.: {fmt([r['price_low'] for r in runs])} €")
        print(f"  Prezzo/notte alta stag.:  {fmt([r['price_high'] for r in runs])} €")
        for key, label in (
            ("offered_at_max_before_fraction", "Notti-offerta future gia al massimo"),
            ("offered_changed_fraction", "Notti-offerta future modificate"),
            ("sold_at_zero_fraction", "Notti vendute a prezzo zero"),
            ("arrival_at_max_fraction", "Coppie categoria-data al massimo all'arrivo"),
            ("demand_clipped_fraction", "Domanda pre-azione oltre 1"),
            ("reward_zero_fraction", "Giorni con reward zero"),
        ):
            print(f"  {label}: {fmt([run['diagnostic_summary'][key] * 100 for run in runs])} %")
        for key in (
            "demand_p05",
            "demand_p50",
            "demand_p95",
            "reward_p05_eur",
            "reward_p50_eur",
            "reward_p95_eur",
            "reward_std_eur",
        ):
            print(
                f"  {key} (media/std tra episodi): "
                f"{fmt([run['diagnostic_summary'][key] for run in runs], digits=6)}"
            )
        for c in range(len(ROOMS)):
            print(
                f"  Occupazione cat. {c + 1}:     {fmt([r['occupancy'][c] * 100 for r in runs])} %"
            )
        for day in SNAPSHOT_DAYS:
            levels = [r["bar_snapshots"][day][0] for r in runs if day in r["bar_snapshots"]]
            at_max = [r["bar_snapshots"][day][1] * 100 for r in runs if day in r["bar_snapshots"]]
            if not levels:
                continue
            print(
                f"  Giorno {day:>3}: BAR medio prossimi {LOOKAHEAD}gg = {fmt(levels)}, "
                f"al massimo = {fmt(at_max)} %"
            )

    write_paired_comparisons(results, output_dir)
    print(f"\nCSV diagnostici e azioni salvati in {output_dir.resolve()}")


def write_paired_comparisons(results: dict[str, list[dict]], output_dir: Path) -> None:
    baseline_name = "theta=0 (sale piano)"
    print(f"\nDifferenze appaiate rispetto a {baseline_name} (EUR):")
    with (output_dir / "paired_differences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", "baseline", "seed", "metric", "difference_eur"])
        for name, runs in results.items():
            if name == baseline_name:
                continue
            for metric in ("revenue", "discounted_revenue"):
                differences = paired_differences(runs, results[baseline_name], metric)
                std = float(differences.std(ddof=1)) if differences.size > 1 else float("nan")
                print(
                    f"  {name}, {metric}: delta medio={differences.mean():,.0f}; "
                    f"std delta={std:,.0f}; "
                    f"seed migliori={np.count_nonzero(differences > 0)}/{differences.size}"
                )
                writer.writerows(
                    [name, baseline_name, run["seed"], metric, difference]
                    for run, difference in zip(runs, differences, strict=True)
                )


if __name__ == "__main__":
    main()
