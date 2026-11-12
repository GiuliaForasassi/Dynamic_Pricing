"""Typed configuration for a single PPO experiment."""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pricing_advertising.utils.constants import (
    LOGISTIC_ACCEPTANCE_AVG_PRICES,
    NUM_WINDOWS,
    WINDOW_SIZE,
)

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
Seed = Annotated[int, Field(strict=True, ge=0, lt=2**32)]


class ConfigSection(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(ConfigSection):
    run_name: Annotated[str, Field(min_length=1)]
    seeds: Annotated[list[Seed], Field(min_length=1)]
    bar_table_path: Path
    tracking_uri: str = "sqlite:///mlflow.db"
    use_subprocesses: Annotated[bool, Field(strict=True)] = True


class SimulationConfig(ConfigSection):
    start_simulation: date
    end_simulation: date
    num_category1_rooms: NonNegativeInt
    num_category2_rooms: NonNegativeInt
    num_category3_rooms: NonNegativeInt
    avg_num_requests: PositiveInt
    warmup_days: NonNegativeInt = 0
    allow_category_alternative: Annotated[bool, Field(strict=True)] = True
    allow_period_alternative: Annotated[bool, Field(strict=True)] = True

    @model_validator(mode="after")
    def validate_dates(self) -> Self:
        if self.start_simulation > self.end_simulation:
            raise ValueError("start_simulation must not be after end_simulation")
        if self.num_category1_rooms + self.num_category2_rooms + self.num_category3_rooms == 0:
            raise ValueError("at least one room is required")
        try:
            self.start_simulation - timedelta(days=self.warmup_days)
            self.end_simulation + timedelta(days=NUM_WINDOWS * WINDOW_SIZE + 1)
        except OverflowError as error:
            raise ValueError("warmup or lookahead exceeds the supported date range") from error
        return self


class PPOConfig(ConfigSection):
    total_timesteps: PositiveInt
    n_envs: PositiveInt
    n_steps: PositiveInt
    learning_rate: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
    gamma: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
    n_epochs: PositiveInt
    batch_size: Annotated[int, Field(strict=True, gt=1)]
    ent_coef: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)] = 0.0
    gae_lambda: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
    clip_range: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
    net_arch: list[PositiveInt] | dict[str, list[PositiveInt]] | None = None
    policy: Literal["MultiInputPolicy"] = "MultiInputPolicy"

    @model_validator(mode="after")
    def validate_rollout(self) -> Self:
        if self.n_steps * self.n_envs <= 1:
            raise ValueError("n_steps * n_envs must be greater than 1")
        if isinstance(self.net_arch, dict) and set(self.net_arch) != {"pi", "vf"}:
            raise ValueError("net_arch must contain exactly pi and vf")
        return self


class EvaluationConfig(ConfigSection):
    eval_frequency: PositiveInt
    eval_seeds: Annotated[list[Seed], Field(min_length=1)]
    price_plot_request_dates: list[date] = Field(default_factory=list)
    evaluate_baselines: Annotated[bool, Field(strict=True)] = True


class CancellationConfig(ConfigSection):
    base_daily_cancellation_prob: Annotated[
        float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)
    ] = 0.005
    timing_strength: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)] = 0.3
    timing_decay_days: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] = 3.0
    high_season_multiplier: Annotated[
        float, Field(strict=True, gt=0, le=1, allow_inf_nan=False)
    ] = 0.8


class ExperimentConfig(ConfigSection):
    run: RunConfig
    simulation: SimulationConfig
    ppo: PPOConfig
    evaluation: EvaluationConfig
    cancellation: CancellationConfig | None

    @model_validator(mode="after")
    def validate_experiment(self) -> Self:
        seeds = self.run.seeds
        eval_seeds = self.evaluation.eval_seeds
        if len(set(seeds)) != len(seeds) or len(set(eval_seeds)) != len(eval_seeds):
            raise ValueError("training and evaluation seed lists must each contain distinct seeds")
        for seed in seeds:
            if seed + self.ppo.n_envs > 2**32:
                raise ValueError("derived training seeds must be in [0, 2**32)")
            if any(seed <= eval_seed < seed + self.ppo.n_envs for eval_seed in eval_seeds):
                raise ValueError("evaluation seeds must not overlap derived training seeds")
        dates = self.evaluation.price_plot_request_dates
        if len(set(dates)) != len(dates):
            raise ValueError("price_plot_request_dates must be distinct")
        if any(
            not self.simulation.start_simulation <= day <= self.simulation.end_simulation
            for day in dates
        ):
            raise ValueError("price_plot_request_dates must be within the simulation dates")
        return self

    def summary(self) -> dict[str, int | float]:
        episode_len = (self.simulation.end_simulation - self.simulation.start_simulation).days + 1
        rollout_size = self.ppo.n_steps * self.ppo.n_envs
        n_rollouts = (self.ppo.total_timesteps + rollout_size - 1) // rollout_size
        effective_timesteps = n_rollouts * rollout_size
        return {
            "n_eval_envs": len(self.evaluation.eval_seeds),
            "n_eval_episodes": len(self.evaluation.eval_seeds),
            "episode_len": episode_len,
            "rollout_size": rollout_size,
            "n_rollouts": n_rollouts,
            "effective_timesteps": effective_timesteps,
            "expected_logged_episodes_per_env": effective_timesteps
            / (episode_len * self.ppo.n_envs),
            "effective_eval_frequency": self.evaluation.eval_frequency * self.ppo.n_envs,
            "scheduled_evaluations": effective_timesteps
            // (self.evaluation.eval_frequency * self.ppo.n_envs),
        }

    def warnings(self) -> list[str]:
        rollout_size = self.ppo.n_steps * self.ppo.n_envs
        messages = []
        if rollout_size % self.ppo.batch_size:
            messages.append(
                "batch_size does not divide the rollout size; the final minibatch will be smaller"
            )
        if self.summary()["scheduled_evaluations"] == 0:
            messages.append("no evaluation checkpoint is scheduled during training")
        return messages

    def runner_kwargs(self) -> dict[str, Any]:
        evaluation = self.evaluation.model_dump()
        evaluation["price_plot_request_dates"] = [
            (day.year, day.month, day.day) for day in self.evaluation.price_plot_request_dates
        ]
        return {
            "run_name": self.run.run_name,
            "seeds": self.run.seeds,
            "use_subprocesses": self.run.use_subprocesses,
            **self.simulation.model_dump(),
            **self.ppo.model_dump(),
            **evaluation,
            "cancellation_config": self.cancellation.model_dump()
            if self.cancellation is not None
            else None,
        }


def load_config(path: Path) -> ExperimentConfig:
    path = path.resolve()
    with path.open(encoding="utf-8") as config_file:
        values = json.load(config_file)
    config = ExperimentConfig.model_validate(values)
    if not config.run.bar_table_path.is_absolute():
        config.run.bar_table_path = (path.parent / config.run.bar_table_path).resolve()
    return config
