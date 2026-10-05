from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.plots.acceptance_rate_tracker import AcceptanceRateTracker
from pricing_advertising.plots.eval_artifact_plots import log_eval_artifact_plots
from pricing_advertising.plots.occupancy_tracker import OccupancyTracker
from pricing_advertising.plots.price_tracker import PriceTracker
from pricing_advertising.plots.request_tracker import RequestTracker
from pricing_advertising.plots.reward_tracker import RewardTracker


# -------------------------- HELPER FUNCTIONS --------------------------
# Computes overall occupancy percentage for a given day across all room categories
def _compute_overall_occupancy_pct(
    occupancy: np.ndarray, day_idx: int, num_rooms: list[int]
) -> float:
    total_rooms = max(1, sum(num_rooms))
    occupied = float(occupancy[:, day_idx].sum())
    return occupied / total_rooms * 100.0


# Computes request metrics (number of requests and mean t_reservation offset)
def _compute_request_metrics(requests: list[Any], start_sim_date: Any) -> tuple[int, float]:  # noqa: ANN401
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


# ----------------------------------------- TRAIN PLOT CALLBACK -----------------------------------------
# Logs training-time occupancy, request, reward, and price metrics to mlflow (tracks a single env only)
class PlotCallback(BaseCallback):
    def __init__(self, log_to_mlflow: bool = True, track_env_idx: int = 0) -> None:
        super().__init__()
        self.log_to_mlflow = log_to_mlflow
        self.track_env_idx = int(track_env_idx)

        # - Initialize trackers for occupancy, requests, rewards, prices, and acceptance rate
        self._occupancy_tracker = OccupancyTracker(split="train", log_to_mlflow=self.log_to_mlflow)
        self._request_tracker = RequestTracker(split="train", log_to_mlflow=self.log_to_mlflow)
        self._reward_tracker = RewardTracker(split="train", log_to_mlflow=self.log_to_mlflow)
        self._price_tracker = PriceTracker(split="train", log_to_mlflow=self.log_to_mlflow)
        self._acceptance_rate_tracker = AcceptanceRateTracker(
            split="train", log_to_mlflow=self.log_to_mlflow
        )
        # Cache for the number of rooms per category in each environment
        self._num_rooms_cache: dict[int, list[int]] = {}

    # -------------------------- Helper functions for SubprocVecEnv --------------------------

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

    # -------------------------- ON STEP --------------------------
    def _on_step(self) -> bool:
        # Get the number of environments
        n_envs = int(self.training_env.num_envs)
        # Determine which environment to track
        env_idx = self.track_env_idx  # default: track env 0
        # Validate the environment index
        if env_idx < 0 or env_idx >= n_envs:
            return True

        # - Extract the metrics for the tracked environment from infos
        metrics = self.locals["infos"][env_idx]["step_metrics"]

        # - Record the extracted metrics for various trackers
        # Each tracker records the relevant metrics for each day
        self._occupancy_tracker.record_day(
            day_index=int(metrics["day_index"]),
            simulation_days=int(metrics["simulation_days"]),
            occupancy_pct=float(metrics["occupancy_pct"]),
        )
        self._request_tracker.record_day(
            day_index=int(metrics["day_index"]),
            simulation_days=int(metrics["simulation_days"]),
            num_requests=int(metrics["num_requests"]),
            request_stays=metrics["request_stays"],
        )
        self._reward_tracker.record_day(
            day_index=int(metrics["day_index"]),
            simulation_days=int(metrics["simulation_days"]),
            reward=float(metrics["reward"]),
        )

        self._price_tracker.record_day(
            day_index=int(metrics["day_index"]),
            start_simulation=metrics["start_simulation"],
            simulation_days=int(metrics["simulation_days"]),
            confirmed_requests=metrics["confirmed_requests"],
            confirmed_nightly_prices=metrics["confirmed_nightly_prices"],
        )

        self._acceptance_rate_tracker.record_day(
            day_index=int(metrics["day_index"]),
            simulation_days=int(metrics["simulation_days"]),
            num_requests=int(metrics["num_requests"]),
            num_offers=int(metrics["num_offers"]),
            num_price_accepted=int(metrics["num_price_accepted"]),
            num_confirmed=int(metrics["num_confirmed"]),
        )
        # - End episode detection and logging
        dones = self.locals.get("dones")
        if dones is not None and dones[env_idx].item():
            self._occupancy_tracker.end_episode()
            self._occupancy_tracker.log_step_metrics(step=self.num_timesteps)

            self._request_tracker.end_episode()
            self._request_tracker.log_step_metrics(step=self.num_timesteps)

            self._reward_tracker.end_episode()
            self._reward_tracker.log_step_metrics(step=self.num_timesteps)

            self._price_tracker.end_episode()
            self._price_tracker.log_step_metrics(step=self.num_timesteps)

            self._acceptance_rate_tracker.end_episode()
            self._acceptance_rate_tracker.log_step_metrics(step=self.num_timesteps)

        return True

    # -------------------------- ON TRAINING END --------------------------
    # Call at the end of training to finalize all trackers
    def _on_training_end(self) -> None:
        self._occupancy_tracker.finalize_training()
        self._request_tracker.finalize_training()
        self._reward_tracker.finalize_training()
        self._price_tracker.finalize_training()
        self._acceptance_rate_tracker.finalize_training()


# -------------------------- EVAL PLOT CALLBACK --------------------------
class EvalOccupancyCallback(EvalCallback):
    def __init__(
        self,
        eval_env: Any,  # noqa: ANN401
        n_eval_episodes: int,  # Number of evaluation episodes to run
        eval_freq: int,  # Frequency of evaluation during training
        log_path: str | None = None,  # Path to save evaluation logs
        best_model_save_path: str | None = None,  # Path to save the best model during evaluation
        deterministic: bool = True,  # Whether to use deterministic actions during evaluation
        render: bool = False,  # Whether to render the environment during evaluation
        verbose: int = 1,  # Verbosity level
        warn: bool = True,  # Whether to show warnings
        log_to_mlflow: bool = True,  # Whether to log evaluation metrics to MLflow
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

        # - Initialize trackers for evaluation metrics
        self._occupancy_tracker = OccupancyTracker(split="val", log_to_mlflow=self.log_to_mlflow)
        self._request_tracker = RequestTracker(split="val", log_to_mlflow=self.log_to_mlflow)
        self._reward_tracker = RewardTracker(split="val", log_to_mlflow=self.log_to_mlflow)
        self._price_tracker = PriceTracker(split="val", log_to_mlflow=self.log_to_mlflow)
        self._acceptance_rate_tracker = AcceptanceRateTracker(
            split="val", log_to_mlflow=self.log_to_mlflow
        )
        # Cache for the number of rooms per category for each environment index
        self._num_rooms_cache: dict[int, list[int]] = {}

        # Artifact tracking for evaluation steps: keeps track of the current checkpoint, episode IDs, and steps for each environment index
        self._artifact_checkpoint: int | None = None
        self._artifact_episode_ids: dict[int, int] = {}
        self._artifact_episode_steps: dict[int, list[dict[str, Any]]] = {}

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

    # Called by evaluate_policy() (invoked internally by EvalCallback._on_step) after every step of evaluation.
    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        super()._log_success_callback(locals_, globals_)

        # - Record the current evaluation step as an artifact.
        self._record_eval_artifact_step(locals_)

        # - Extract the step metrics from the locals dictionary.
        metrics = locals_["info"]["step_metrics"]

        # - Record the various metrics for the current evaluation step in the respective trackers.
        self._occupancy_tracker.record_day(
            day_index=int(metrics["day_index"]),
            simulation_days=int(metrics["simulation_days"]),
            occupancy_pct=float(metrics["occupancy_pct"]),
        )
        self._request_tracker.record_day(
            day_index=int(metrics["day_index"]),
            simulation_days=int(metrics["simulation_days"]),
            num_requests=int(metrics["num_requests"]),
            request_stays=metrics["request_stays"],
        )
        self._reward_tracker.record_day(
            day_index=int(metrics["day_index"]),
            simulation_days=int(metrics["simulation_days"]),
            reward=float(metrics["reward"]),
        )

        self._price_tracker.record_day(
            day_index=int(metrics["day_index"]),
            start_simulation=metrics["start_simulation"],
            simulation_days=int(metrics["simulation_days"]),
            confirmed_requests=metrics["confirmed_requests"],
            confirmed_nightly_prices=metrics["confirmed_nightly_prices"],
        )

        self._acceptance_rate_tracker.record_day(
            day_index=int(metrics["day_index"]),
            simulation_days=int(metrics["simulation_days"]),
            num_requests=int(metrics["num_requests"]),
            num_offers=int(metrics["num_offers"]),
            num_price_accepted=int(metrics["num_price_accepted"]),
            num_confirmed=int(metrics["num_confirmed"]),
        )

        # - If the evaluation episode is done, signal the end of the episode to all trackers.
        if bool(locals_["done"]):
            self._occupancy_tracker.end_episode()

            self._request_tracker.end_episode()

            self._reward_tracker.end_episode()

            self._price_tracker.end_episode()

            self._acceptance_rate_tracker.end_episode()

    # ------------------------------- ON STEP -------------------------------
    def _on_step(self) -> bool:
        # - Save the previous best mean reward
        previous_best = self.best_mean_reward
        # Call the parent class's _on_step() method to continue training
        continue_training = super()._on_step()

        # - If the best mean reward has improved, save the corresponding model checkpoint.
        if self.best_mean_reward > previous_best and self.best_model_save_path is not None:
            # Save the normalization statistics
            normalization = self.model.get_vec_normalize_env()
            if normalization is not None:
                normalization.save(str(Path(self.best_model_save_path) / "best_vecnormalize.pkl"))

        # - Log the step metrics for all trackers
        self._request_tracker.log_step_metrics(step=self.num_timesteps)
        self._acceptance_rate_tracker.log_step_metrics(step=self.num_timesteps)
        self._reward_tracker.log_step_metrics(step=self.num_timesteps)
        self._occupancy_tracker.log_step_metrics(step=self.num_timesteps)
        self._price_tracker.log_step_metrics(step=self.num_timesteps)

        return continue_training

    # ------------------------------- ON TRAINING END -------------------------------
    def _on_training_end(self) -> None:
        super()._on_training_end()
        # Finalize training for all trackers
        self._occupancy_tracker.finalize_training()
        self._request_tracker.finalize_training()
        self._reward_tracker.finalize_training()
        self._price_tracker.finalize_training()
        self._acceptance_rate_tracker.finalize_training()

    # ------------------------------- RECORD EVAL ARTIFACT STEP -------------------------------
    def _record_eval_artifact_step(
        self,
        locals_: dict[str, Any],  # Local variables from the evaluation environment
    ) -> None:
        if not self.log_to_mlflow:
            return

        # Extract the current checkpoint and environment index from the local variables
        checkpoint = int(self.num_timesteps)
        env_idx = int(locals_["i"])

        # Check
        if checkpoint != self._artifact_checkpoint:
            if any(self._artifact_episode_steps.values()):
                raise RuntimeError("Change of checkpoint with unclosed evaluation data.")
            # Update the artifact checkpoint and clear previous episode data
            self._artifact_checkpoint = checkpoint
            self._artifact_episode_ids.clear()
            self._artifact_episode_steps.clear()

        # Extract the evaluation artifacts from the local variables
        payload = locals_["info"].get("eval_artifacts")
        if payload is None:
            raise RuntimeError(
                "Missing eval_artifacts data: check the evaluation environment configuration."
            )
        # Append the current evaluation step to the list of steps for this environment index.
        steps = self._artifact_episode_steps.setdefault(env_idx, [])
        steps.append(payload)

        # If the episode is not done, return early.
        if not bool(locals_["done"]):
            return

        # Retrieve the episode ID for the current environment index, defaulting to 0 if not found.
        episode_id = self._artifact_episode_ids.get(env_idx, 0)

        # Retrieve the snapshot dates for the current environment index.
        snapshot_dates = self.eval_env.get_attr(
            "artifact_snapshot_dates",
            indices=env_idx,
        )[0]

        # Construct the artifact dictionary containing evaluation data for logging.
        artifact = {
            "schema_version": 3,  # Version of the artifact schema
            "decision_end_inclusive": True,  # Indicates whether the decision at the end of the episode is inclusive
            "checkpoint_timesteps": checkpoint,  # Number of timesteps at the current checkpoint
            "env_index": env_idx,  # Index of the environment within the evaluation set
            "episode_id": episode_id,  # ID of the current episode within the environment
            "snapshot_dates": sorted(
                value.isoformat() for value in snapshot_dates
            ),  # List of snapshot dates for the current episode
            "steps": steps,  # List of evaluation steps for the current episode
        }

        # Log the artifact dictionary to MLflow as a JSON file.
        mlflow.log_dict(
            artifact,
            (
                f"eval_data/{checkpoint:08d}_timesteps/"
                f"env_{env_idx:02d}/episode_{episode_id:05d}.json"
            ),
        )

        # Generate and log evaluation artifact plots based on the current artifact.
        log_eval_artifact_plots(artifact)

        # Clear the evaluation steps for the current environment index and increment the episode ID.
        self._artifact_episode_steps[env_idx] = []
        self._artifact_episode_ids[env_idx] = episode_id + 1
