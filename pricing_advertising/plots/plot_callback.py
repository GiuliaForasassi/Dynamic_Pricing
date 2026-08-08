from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.plots.acceptance_rate_tracker import AcceptanceRateTracker
from pricing_advertising.plots.occupancy_tracker import OccupancyTracker
from pricing_advertising.plots.price_tracker import PriceTracker
from pricing_advertising.plots.request_tracker import RequestTracker
from pricing_advertising.plots.reward_tracker import RewardTracker


######################## Calculates the overall occupancy rate ####################
def _compute_overall_occupancy_pct(
    occupancy: np.ndarray, day_idx: int, num_rooms: list[int]
) -> float:
    """Occupancy percentage for `day_idx`, aggregated across all room categories."""
    total_rooms = max(1, sum(num_rooms))
    occupied = float(occupancy[:, day_idx].sum())
    return occupied / total_rooms * 100.0


######################## Calculates the requests metrics #########################
def _compute_request_metrics(requests: list[Any], start_sim_date: Any) -> tuple[int, float]:  # noqa: ANN401
    """Compute request count and mean t_reservation (as day offset from sim start) from requests.

    Args:
        requests: List of Request objects, each with a t_reservation attribute (a date).
        start_sim_date: The simulation start date.

    Returns:
        (num_requests, t_reservation_mean) where t_reservation_mean is the mean
        offset (in days) of check-in dates from simulation start. If no requests,
        returns (0, 0.0).
    """
    num_requests = len(requests)
    if num_requests == 0:
        return 0, 0.0

    # Extract t_reservation values and convert to day offsets from simulation start
    t_reservation_offsets = []
    for req in requests:
        offset = float((req.t_reservation - start_sim_date).days)
        t_reservation_offsets.append(offset)

    mean_t_reservation = float(np.mean(t_reservation_offsets))
    return num_requests, mean_t_reservation


#################################### TRAIN PLOT CALLBACK ##########################################
class PlotCallback(BaseCallback):
    """Logs training-time occupancy, request, reward, and price metrics to mlflow (tracks a single env only)."""

    def __init__(self, log_to_mlflow: bool = True, track_env_idx: int = 0) -> None:
        super().__init__()
        self.log_to_mlflow = log_to_mlflow
        self.track_env_idx = int(track_env_idx)

        self._occupancy_tracker = OccupancyTracker(split="train", log_to_mlflow=self.log_to_mlflow)
        self._request_tracker = RequestTracker(split="train", log_to_mlflow=self.log_to_mlflow)
        self._reward_tracker = RewardTracker(split="train", log_to_mlflow=self.log_to_mlflow)
        self._price_tracker = PriceTracker(split="train", log_to_mlflow=self.log_to_mlflow)
        self._acceptance_rate_tracker = AcceptanceRateTracker(
            split="train", log_to_mlflow=self.log_to_mlflow
        )
        self._num_rooms_cache: dict[int, list[int]] = {}
        self._price_tracker_initialized = False

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

        # Initialize price tracker on first step
        if not self._price_tracker_initialized:
            start_sim = self._get_attr("start_simulation", env_idx)
            end_sim = self._get_attr("end_simulation", env_idx)
            simulation_days = int((end_sim - start_sim).days)
            self._price_tracker.set_simulation_params(start_sim, simulation_days)
            self._price_tracker_initialized = True

        # ---- Retrieve simulation state ----
        # Note: env increments `today` at the end of step(), so `today` is already the next day
        # We need to track metrics for the day that just completed (today - 1 day)
        today = self._get_attr("today", env_idx)
        start_sim = self._get_attr("start_simulation", env_idx)
        completed_day_index = int((today - start_sim).days) - 1

        num_rooms = self._get_num_rooms_by_cat(env_idx)
        occupancy = self._get_attr("occupancy", env_idx)
        occ_pct = _compute_overall_occupancy_pct(occupancy, completed_day_index, num_rooms)
        self._occupancy_tracker.record_day(occ_pct)

        # ---- Record request metrics ----
        request_generator = self._get_attr("request_generator", env_idx)
        requests = request_generator.fetch_requests_for_date(today)
        num_requests, t_reservation_mean = _compute_request_metrics(requests, start_sim)
        self._request_tracker.record_day(num_requests, t_reservation_mean)

        # ---- Record reward metrics ----
        rewards = self.locals.get("rewards")
        if rewards is not None and env_idx < len(rewards):
            reward = float(rewards[env_idx])
            self._reward_tracker.record_day(reward)

        # ---- Record price metrics ----
        final_requests = self._get_attr("last_final_requests", env_idx)
        final_prices = self._get_attr("last_final_prices", env_idx)
        self._price_tracker.record_request_prices(final_requests, final_prices)

        # ---- Record acceptance rate metrics ----
        accepted_by_day = self._get_attr("accepted_by_day", env_idx)
        rejected_by_day = self._get_attr("rejected_by_day", env_idx)
        total_accepted = float(accepted_by_day[:, completed_day_index].sum())
        total_rejected = float(rejected_by_day[:, completed_day_index].sum())
        if total_accepted + total_rejected > 0:
            acceptance_rate = total_accepted / (total_accepted + total_rejected)
            self._acceptance_rate_tracker.record_day(acceptance_rate)

        # ---- end episode detection ----
        dones = self.locals.get("dones")
        if dones is not None and dones[env_idx].item():
            mean_occ = self._occupancy_tracker.end_episode()
            if mean_occ is not None:
                self._occupancy_tracker.log_step_metric(float(mean_occ), step=self.num_timesteps)

            self._request_tracker.end_episode()

            mean_reward = self._reward_tracker.end_episode()
            if mean_reward is not None:
                self._reward_tracker.log_step_metric(float(mean_reward), step=self.num_timesteps)

            mean_price = self._price_tracker.end_episode()
            if mean_price is not None:
                self._price_tracker.log_step_metric(float(mean_price), step=self.num_timesteps)

            mean_acceptance_rate = self._acceptance_rate_tracker.end_episode()
            if mean_acceptance_rate is not None:
                self._acceptance_rate_tracker.log_step_metric(
                    float(mean_acceptance_rate), step=self.num_timesteps
                )

        return True

    # ========== ON TRAINING ==========
    def _on_training_end(self) -> None:
        self._occupancy_tracker.finalize_training()
        self._request_tracker.finalize_training()
        self._reward_tracker.finalize_training()
        self._price_tracker.finalize_training()
        self._acceptance_rate_tracker.finalize_training()


#################################### EVAL PLOT CALLBACK ##########################################
class EvalOccupancyCallback(EvalCallback):
    """EvalCallback that additionally logs validation occupancy, request, reward, and price metrics to mlflow."""

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

        self._occupancy_tracker = OccupancyTracker(split="val", log_to_mlflow=self.log_to_mlflow)
        self._request_tracker = RequestTracker(split="val", log_to_mlflow=self.log_to_mlflow)
        self._reward_tracker = RewardTracker(split="val", log_to_mlflow=self.log_to_mlflow)
        self._price_tracker = PriceTracker(split="val", log_to_mlflow=self.log_to_mlflow)
        self._acceptance_rate_tracker = AcceptanceRateTracker(
            split="val", log_to_mlflow=self.log_to_mlflow
        )
        self._num_rooms_cache: dict[int, list[int]] = {}
        self._price_tracker_initialized = False
        # Per-episode means collected during the eval round currently in progress.
        self._round_occ_means: list[float] = []
        self._round_num_requests_means: list[float] = []
        self._round_t_reservation_means: list[float] = []
        self._round_reward_means: list[float] = []
        self._round_price_means: list[float] = []
        self._round_acceptance_rate_means: list[float] = []

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

        # Initialize price tracker on first step
        if not self._price_tracker_initialized:
            start_sim = env.get_attr("start_simulation", indices=env_idx)[0]
            end_sim = env.get_attr("end_simulation", indices=env_idx)[0]
            simulation_days = int((end_sim - start_sim).days)
            self._price_tracker.set_simulation_params(start_sim, simulation_days)
            self._price_tracker_initialized = True

        # ---- Retrieve simulation state ----
        # Note: env increments `today` at the end of step(), so `today` is already the next day
        # We need to track metrics for the day that just completed (today - 1 day)
        today = env.get_attr("today", indices=env_idx)[0]
        start_sim = env.get_attr("start_simulation", indices=env_idx)[0]
        completed_day_index = int((today - start_sim).days) - 1

        num_rooms = self._get_num_rooms_by_cat(env, env_idx)
        occupancy = env.get_attr("occupancy", indices=env_idx)[0]
        occ_pct = _compute_overall_occupancy_pct(occupancy, completed_day_index, num_rooms)
        self._occupancy_tracker.record_day(occ_pct)

        # ---- Record request metrics ----
        request_generator = env.get_attr("request_generator", indices=env_idx)[0]  # TODO slow
        requests = request_generator.fetch_requests_for_date(today)
        num_requests, t_reservation_mean = _compute_request_metrics(requests, start_sim)
        self._request_tracker.record_day(num_requests, t_reservation_mean)

        # ---- Record reward metrics ----
        reward = locals_.get("reward", 0.0)
        self._reward_tracker.record_day(float(reward))

        # ---- Record price metrics ----
        final_requests = env.get_attr("last_final_requests", indices=env_idx)[0]
        final_prices = env.get_attr("last_final_prices", indices=env_idx)[0]
        self._price_tracker.record_request_prices(final_requests, final_prices)

        # ---- Record acceptance rate metrics ----
        accepted_by_day = env.get_attr("accepted_by_day", indices=env_idx)[0]
        rejected_by_day = env.get_attr("rejected_by_day", indices=env_idx)[0]
        total_accepted = float(accepted_by_day[:, completed_day_index].sum())
        total_rejected = float(rejected_by_day[:, completed_day_index].sum())
        if total_accepted + total_rejected > 0:
            acceptance_rate = total_accepted / (total_accepted + total_rejected)
            self._acceptance_rate_tracker.record_day(acceptance_rate)

        if bool(locals_["done"]):
            mean_occ = self._occupancy_tracker.end_episode()
            if mean_occ is not None:
                self._round_occ_means.append(mean_occ)

            self._request_tracker.end_episode()

            mean_reward = self._reward_tracker.end_episode()
            if mean_reward is not None:
                self._round_reward_means.append(mean_reward)

            mean_price = self._price_tracker.end_episode()
            if mean_price is not None:
                self._round_price_means.append(mean_price)

            mean_acceptance_rate = self._acceptance_rate_tracker.end_episode()
            if mean_acceptance_rate is not None:
                self._round_acceptance_rate_means.append(mean_acceptance_rate)

    # Fires once per eval round (every `eval_freq` training steps): SB3's
    # EvalCallback._on_step() only calls evaluate_policy() (which drives
    # `_log_success_callback` above) on those steps, so `_round_means` only
    # has content right after a round completes.
    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self._round_occ_means:
            self._occupancy_tracker.log_step_metric(
                float(np.mean(self._round_occ_means)), step=self.num_timesteps
            )
            self._round_occ_means = []

        if self._round_reward_means:
            self._reward_tracker.log_step_metric(
                float(np.mean(self._round_reward_means)), step=self.num_timesteps
            )
            self._round_reward_means = []

        if self._round_price_means:
            self._price_tracker.log_step_metric(
                float(np.mean(self._round_price_means)), step=self.num_timesteps
            )
            self._round_price_means = []

        if self._round_acceptance_rate_means:
            self._acceptance_rate_tracker.log_step_metric(
                float(np.mean(self._round_acceptance_rate_means)), step=self.num_timesteps
            )
            self._round_acceptance_rate_means = []

        return continue_training

    def _on_training_end(self) -> None:
        super()._on_training_end()
        self._occupancy_tracker.finalize_training()
        self._request_tracker.finalize_training()
        self._reward_tracker.finalize_training()
        self._price_tracker.finalize_training()
        self._acceptance_rate_tracker.finalize_training()
