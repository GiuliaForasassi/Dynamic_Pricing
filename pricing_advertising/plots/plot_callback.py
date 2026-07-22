from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.plots.occupancy_tracker import OccupancyTracker


def _compute_overall_occupancy_pct(
    occupancy: np.ndarray, day_idx: int, num_rooms: list[int]
) -> float:
    """Occupancy percentage for `day_idx`, aggregated across all room categories."""
    total_rooms = max(1, sum(num_rooms))
    occupied = float(occupancy[:, day_idx].sum())
    return occupied / total_rooms * 100.0


class PlotCallback(BaseCallback):
    """Logs training-time occupancy metrics to mlflow (tracks a single env only)."""

    def __init__(self, log_to_mlflow: bool = True, track_env_idx: int = 0) -> None:
        super().__init__()
        self.log_to_mlflow = log_to_mlflow
        self.track_env_idx = int(track_env_idx)

        self._tracker = OccupancyTracker(split="train", log_to_mlflow=self.log_to_mlflow)
        self._num_rooms_cache: dict[int, list[int]] = {}

    # ========== HELPER FUNCTIONS FOR SubprocVecEnv ==========

    # Returns the "name" attribute of the env index idx
    def _get_attr(self, name: str, idx: int) -> Any:  # noqa: ANN401
        return self.training_env.get_attr(name, indices=idx)[0]

    # Calls a method (defined in env) in the subprocess
    def _env_method(self, method_name: str, *args: Any, idx: int, **kwargs: Any) -> Any:  # noqa: ANN401
        return self.training_env.env_method(method_name, *args, indices=idx, **kwargs)[0]

    # Returns [rooms_cat1, rooms_cat2, rooms_cat3] for env idx, with caching
    def _get_num_rooms_by_cat(self, idx: int) -> list[int]:
        if idx in self._num_rooms_cache:
            return self._num_rooms_cache[idx]
        num_cats: int = int(self._get_attr("num_room_categories", idx))
        out: list[int] = []
        for c in range(1, num_cats + 1):
            n = self._env_method("get_num_rooms_per_category", RoomCategory(c), idx=idx)
            out.append(int(n))
        self._num_rooms_cache[idx] = out
        return out

    # ========== ON STEP ==========
    def _on_step(self) -> bool:
        n_envs = int(self.training_env.num_envs)
        env_idx = self.track_env_idx  # default: track env 0
        if env_idx < 0 or env_idx >= n_envs:
            return True

        # ---- I retrieve the attributes of the env I am considering ----
        today = self._get_attr("today", env_idx)
        start_sim = self._get_attr("start_simulation", env_idx)
        current_day_index = int((today - start_sim).days)

        num_rooms = self._get_num_rooms_by_cat(env_idx)
        occupancy = self._get_attr("occupancy", env_idx)
        occ_pct = _compute_overall_occupancy_pct(occupancy, current_day_index, num_rooms)
        self._tracker.record_day(occ_pct)

        # ---- end episode detection ----
        dones = self.locals.get("dones")
        if dones is not None and bool(np.asarray(dones, dtype=bool).ravel()[env_idx]):
            mean_occ = self._tracker.end_episode()
            if mean_occ is not None:
                self._tracker.log_step_metric(mean_occ, step=self.num_timesteps)

        return True

    # ========== ON TRAINING ==========
    def _on_training_end(self) -> None:
        self._tracker.finalize_training()


class EvalOccupancyCallback(EvalCallback):
    """EvalCallback that additionally logs validation occupancy metrics to mlflow."""

    def __init__(
        self,
        eval_env: Any,  # noqa: ANN401
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str | None = None,
        best_model_save_path: str | None = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        log_to_mlflow: bool = True,
    ) -> None:
        super().__init__(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )
        self.log_to_mlflow = log_to_mlflow

        self._tracker = OccupancyTracker(split="val", log_to_mlflow=self.log_to_mlflow)
        self._num_rooms_cache: dict[int, list[int]] = {}
        # Per-episode means collected during the eval round currently in progress.
        self._round_means: list[float] = []

    def _get_num_rooms_by_cat(self, env: Any, idx: int) -> list[int]:  # noqa: ANN401
        if idx in self._num_rooms_cache:
            return self._num_rooms_cache[idx]
        num_cats: int = int(env.get_attr("num_room_categories", indices=idx)[0])
        out: list[int] = []
        for c in range(1, num_cats + 1):
            n = env.env_method("get_num_rooms_per_category", RoomCategory(c), indices=idx)[0]
            out.append(int(n))
        self._num_rooms_cache[idx] = out
        return out

    # Called by evaluate_policy() (invoked internally by EvalCallback._on_step) after every step
    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        super()._log_success_callback(locals_, globals_)

        env = locals_["env"]
        env_idx = int(locals_["i"])

        today = env.get_attr("today", indices=env_idx)[0]
        start_sim = env.get_attr("start_simulation", indices=env_idx)[0]
        current_day_index = int((today - start_sim).days)

        num_rooms = self._get_num_rooms_by_cat(env, env_idx)
        occupancy = env.get_attr("occupancy", indices=env_idx)[0]
        occ_pct = _compute_overall_occupancy_pct(occupancy, current_day_index, num_rooms)
        self._tracker.record_day(occ_pct)

        if bool(locals_["done"]):
            mean_occ = self._tracker.end_episode()
            if mean_occ is not None:
                self._round_means.append(mean_occ)

    # Fires once per eval round (every `eval_freq` training steps): SB3's
    # EvalCallback._on_step() only calls evaluate_policy() (which drives
    # `_log_success_callback` above) on those steps, so `_round_means` only
    # has content right after a round completes.
    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self._round_means:
            self._tracker.log_step_metric(
                float(np.mean(self._round_means)), step=self.num_timesteps
            )
            self._round_means = []
        return continue_training

    def _on_training_end(self) -> None:
        super()._on_training_end()
        self._tracker.finalize_training()
