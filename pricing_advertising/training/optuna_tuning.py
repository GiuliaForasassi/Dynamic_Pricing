"""Two-stage experimental calibration and PPO tuning, without changing main defaults."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.pricing_model.bar_table import set_bar_table_from_csv
from pricing_advertising.training.random_policy import RandomPolicy
from pricing_advertising.utils.constants import SEEDS


@dataclass(frozen=True)
class Scenario:
    num_category1_rooms: int
    num_category2_rooms: int
    num_category3_rooms: int
    avg_num_requests: int

    @property
    def rooms(self) -> tuple[int, int, int]:
        return self.num_category1_rooms, self.num_category2_rooms, self.num_category3_rooms


@dataclass(frozen=True)
class Settings:
    start: date = date(2025, 1, 1)
    end: date = date(2025, 12, 31)
    target_occupancy: float = 0.7
    scenario_trials: int = 20
    ppo_trials: int = 10
    timesteps: int = 20_480
    n_envs: int = 4
    train_seeds: tuple[int, ...] = tuple(SEEDS[:3])
    calibration_seeds: tuple[int, ...] = tuple(SEEDS[0] + 10_000 + i for i in range(3))
    validation_seeds: tuple[int, ...] = tuple(SEEDS[0] + 20_000 + i for i in range(3))
    holdout_seeds: tuple[int, ...] = tuple(SEEDS[0] + 30_000 + i for i in range(5))
    sampler_seed: int = SEEDS[0]
    smoke: bool = False

    def validate(self) -> None:
        if self.start > self.end or not 0 < self.target_occupancy < 1:
            raise ValueError("Dates and occupancy target must define a valid experiment.")
        for value in (self.scenario_trials, self.ppo_trials, self.timesteps, self.n_envs):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("Trial counts, timesteps and n_envs must be positive integers.")
        groups = (
            self.train_seeds,
            self.calibration_seeds,
            self.validation_seeds,
            self.holdout_seeds,
        )
        if any(not group for group in groups):
            raise ValueError("Each seed split must be nonempty.")
        all_seeds = [
            *(seed + i for seed in self.train_seeds for i in range(self.n_envs)),
            *self.calibration_seeds,
            *self.validation_seeds,
            *self.holdout_seeds,
        ]
        if len(set(all_seeds)) != len(all_seeds):
            raise ValueError(
                "Training, calibration, validation and holdout seeds must be disjoint."
            )
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
            for seed in [*self.train_seeds, *all_seeds, self.sampler_seed]
        ):
            raise ValueError("Seeds must be integers in [0, 2**32).")
        largest_rollout = self.n_envs * (8 if self.smoke else 512)
        if self.timesteps % largest_rollout:
            raise ValueError(f"timesteps must be divisible by {largest_rollout} for equal budgets.")


def make_env(scenario: Scenario, settings: Settings, seed: int) -> PricingEnv:
    return PricingEnv(
        start_simulation=settings.start,
        end_simulation=settings.end,
        hotel=Hotel(*scenario.rooms),
        avg_num_requests=scenario.avg_num_requests,
        seed=seed,
        allow_category_alternative=False,
        allow_period_alternative=False,
    )


def make_training_env(scenario: Scenario, settings: Settings, seed: int) -> Monitor:
    return Monitor(make_env(scenario, settings, seed))


def evaluate(
    scenario: Scenario,
    settings: Settings,
    seeds: tuple[int, ...],
    model: PPO | None = None,
) -> list[dict[str, Any]]:
    """Fresh environment per seed; raw EUR rewards and in-horizon occupancy."""
    results: list[dict[str, Any]] = []
    for seed in seeds:
        env = make_env(scenario, settings, seed)
        try:
            random_policy = RandomPolicy(env.action_space, seed)
            observation, _ = env.reset(seed=seed)
            revenue = 0.0
            done = False
            while not done:
                action, _ = (
                    random_policy.predict(observation)
                    if model is None
                    else model.predict(observation, deterministic=True)
                )
                observation, reward, terminated, truncated, _ = env.step(action)
                revenue += float(reward)
                done = terminated or truncated
            occupancy = env.occupancy[:, : env.total_days].sum(axis=1) / (
                np.asarray(scenario.rooms) * env.total_days
            )
            if not np.isfinite(revenue) or not np.isfinite(occupancy).all():
                raise ValueError("Evaluation produced non-finite revenue or occupancy.")
            results.append({"seed": seed, "revenue": revenue, "occupancy": occupancy.tolist()})
        finally:
            env.close()
    return results


def occupancy_loss(results: list[dict[str, Any]], target: float) -> float:
    # Penalize each category and seed separately: opposite extremes must not cancel.
    return float(np.mean(np.abs(np.asarray([row["occupancy"] for row in results]) - target)))


def sample_scenario(trial: optuna.Trial) -> Scenario:
    return Scenario(
        num_category1_rooms=trial.suggest_int("num_category1_rooms", 20, 200, step=10),
        num_category2_rooms=trial.suggest_int("num_category2_rooms", 10, 150, step=10),
        num_category3_rooms=trial.suggest_int("num_category3_rooms", 10, 100, step=10),
        avg_num_requests=trial.suggest_int("avg_num_requests", 20, 250, step=10),
    )


def sample_ppo(trial: optuna.Trial, smoke: bool) -> dict[str, Any]:
    width = trial.suggest_categorical("net_width", [64] if smoke else [64, 128, 256])
    return {
        "n_steps": trial.suggest_categorical("n_steps", [8] if smoke else [128, 256, 512]),
        "batch_size": trial.suggest_categorical("batch_size", [8] if smoke else [32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.95, 0.9995),
        "n_epochs": trial.suggest_categorical("n_epochs", [1] if smoke else [5, 10]),
        "ent_coef": trial.suggest_categorical("ent_coef", [0.0, 1e-4, 1e-3, 1e-2]),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "policy_kwargs": {"net_arch": {"pi": [width, width], "vf": [width, width]}},
    }


def train_model(
    scenario: Scenario,
    settings: Settings,
    parameters: dict[str, Any],
    seed: int,
    output: Path,
) -> PPO:
    # Same normalization as run_simulation. DummyVecEnv avoids fork and plot/MLflow overhead.
    env = VecNormalize(
        DummyVecEnv(
            [
                partial(make_training_env, scenario, settings, seed + i)
                for i in range(settings.n_envs)
            ]
        ),
        norm_obs=False,
        norm_reward=True,
        gamma=parameters["gamma"],
        clip_reward=10.0,
    )
    try:
        env.seed(seed)
        model = PPO("MultiInputPolicy", env, seed=seed, device="cpu", **parameters)
        model.learn(total_timesteps=settings.timesteps)
        output.mkdir(parents=True, exist_ok=True)
        model.save(output / "model.zip")
        env.save(str(output / "vecnormalize.pkl"))
        return model
    finally:
        env.close()


def run_study(settings: Settings, output: Path, bar_csv: Path) -> dict[str, Any]:
    settings.validate()
    set_bar_table_from_csv(bar_csv)
    # Never mix studies with different data, budgets or search spaces.
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        **asdict(settings),
        "start": settings.start.isoformat(),
        "end": settings.end.isoformat(),
        "bar_sha256": hashlib.sha256(bar_csv.read_bytes()).hexdigest(),
        "allow_category_alternative": False,
        "allow_period_alternative": False,
    }
    (output / "settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    storage = f"sqlite:///{(output / 'studies.sqlite3').resolve()}"
    scenarios = optuna.create_study(
        study_name="scenario",
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=settings.sampler_seed, n_startup_trials=5),
    )
    scenarios.enqueue_trial(asdict(Scenario(100, 50, 30, 60)))

    def scenario_objective(trial: optuna.Trial) -> float:
        results = evaluate(sample_scenario(trial), settings, settings.calibration_seeds)
        trial.set_user_attr("calibration", results)
        return occupancy_loss(results, settings.target_occupancy)

    scenarios.optimize(scenario_objective, n_trials=settings.scenario_trials)
    scenario = Scenario(**scenarios.best_params)
    ppo_study = optuna.create_study(
        study_name="ppo",
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.sampler_seed, n_startup_trials=5),
    )
    if not settings.smoke:
        ppo_study.enqueue_trial(
            {
                "net_width": 64,
                "n_steps": 512,
                "batch_size": 64,
                "learning_rate": 3e-4,
                "gamma": 0.995,
                "n_epochs": 10,
                "ent_coef": 0.0,
                "gae_lambda": 0.95,
                "clip_range": 0.1,
            }
        )
    baseline_validation = evaluate(scenario, settings, settings.validation_seeds)

    def ppo_objective(trial: optuna.Trial) -> float:
        parameters = sample_ppo(trial, settings.smoke)
        trial.set_user_attr("ppo_parameters", parameters)
        results: list[dict[str, Any]] = []
        for seed in settings.train_seeds:
            model = train_model(
                scenario, settings, parameters, seed, output / f"trial_{trial.number}" / str(seed)
            )
            results.append(
                {
                    "train_seed": seed,
                    "episodes": evaluate(scenario, settings, settings.validation_seeds, model),
                }
            )
        trial.set_user_attr("validation", results)
        return float(
            np.mean([episode["revenue"] for run in results for episode in run["episodes"]])
        )

    ppo_study.optimize(ppo_objective, n_trials=settings.ppo_trials)
    baseline_holdout = evaluate(scenario, settings, settings.holdout_seeds)
    holdout: list[dict[str, Any]] = []
    for seed in settings.train_seeds:
        model_path = output / f"trial_{ppo_study.best_trial.number}" / str(seed) / "model.zip"
        model = PPO.load(model_path, device="cpu")
        episodes = evaluate(scenario, settings, settings.holdout_seeds, model)
        holdout.append(
            {
                "train_seed": seed,
                "model_path": str(model_path.resolve()),
                "episodes": episodes,
                "paired_revenue_deltas": [
                    episode["revenue"] - baseline["revenue"]
                    for episode, baseline in zip(episodes, baseline_holdout, strict=True)
                ],
            }
        )
    report = {
        "status": "smoke_only_not_a_recommendation" if settings.smoke else "exploratory",
        "settings": metadata,
        "scenario": asdict(scenario),
        "scenario_occupancy_mae": scenarios.best_value,
        "scenario_calibration": scenarios.best_trial.user_attrs["calibration"],
        "best_ppo_trial": ppo_study.best_trial.number,
        "ppo_parameters": ppo_study.best_trial.user_attrs["ppo_parameters"],
        "validation_mean_revenue": ppo_study.best_value,
        "baseline_validation": baseline_validation,
        "baseline_holdout": baseline_holdout,
        "ppo_holdout": holdout,
        "holdout_mean_paired_delta": float(
            np.mean([delta for run in holdout for delta in run["paired_revenue_deltas"]])
        ),
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    logger.info("Study completed: {} (status: {})", output / "results.json", report["status"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bar-csv", type=Path, default=Path("data/bar_table.csv"))
    parser.add_argument("--scenario-trials", type=int, default=20)
    parser.add_argument("--ppo-trials", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=20_480)
    parser.add_argument(
        "--smoke", action="store_true", help="Tiny plumbing check, not optimization."
    )
    args = parser.parse_args()
    settings = (
        Settings(
            end=date(2025, 1, 7),
            scenario_trials=2,
            ppo_trials=1,
            timesteps=32,
            n_envs=1,
            train_seeds=(SEEDS[0],),
            calibration_seeds=(SEEDS[0] + 10_000,),
            validation_seeds=(SEEDS[0] + 20_000,),
            holdout_seeds=(SEEDS[0] + 30_000, SEEDS[0] + 30_001),
            smoke=True,
        )
        if args.smoke
        else Settings(
            scenario_trials=args.scenario_trials,
            ppo_trials=args.ppo_trials,
            timesteps=args.timesteps,
        )
    )
    output = args.output or Path("tmp/optuna") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_study(settings, output, args.bar_csv)


if __name__ == "__main__":
    main()
