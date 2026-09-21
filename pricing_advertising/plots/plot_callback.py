from __future__ import annotations

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

        metrics = self.locals["infos"][env_idx]["step_metrics"]

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
        # ---- end episode detection ----
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

    # Called by evaluate_policy() (invoked internally by EvalCallback._on_step) after every step
    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        super()._log_success_callback(locals_, globals_)

        # Raccoglie i dati dello step della vera evaluation.
        self._record_eval_artifact_step(locals_)

        metrics = locals_["info"]["step_metrics"]

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

        if bool(locals_["done"]):
            self._occupancy_tracker.end_episode()

            self._request_tracker.end_episode()

            self._reward_tracker.end_episode()

            self._price_tracker.end_episode()

            self._acceptance_rate_tracker.end_episode()

    # Fires once per eval round (every `eval_freq` training steps): SB3's
    # EvalCallback._on_step() only calls evaluate_policy() (which drives
    # `_log_success_callback` above) on those steps, so `_round_means` only
    # has content right after a round completes.
    def _on_step(self) -> bool:
        continue_training = super()._on_step()

        self._request_tracker.log_step_metrics(step=self.num_timesteps)
        self._acceptance_rate_tracker.log_step_metrics(step=self.num_timesteps)
        self._reward_tracker.log_step_metrics(step=self.num_timesteps)
        self._occupancy_tracker.log_step_metrics(step=self.num_timesteps)
        self._price_tracker.log_step_metrics(step=self.num_timesteps)

        return continue_training

    def _on_training_end(self) -> None:
        super()._on_training_end()
        self._occupancy_tracker.finalize_training()
        self._request_tracker.finalize_training()
        self._reward_tracker.finalize_training()
        self._price_tracker.finalize_training()
        self._acceptance_rate_tracker.finalize_training()

    def _record_eval_artifact_step(
        self,
        locals_: dict[str, Any],
    ) -> None:
        if not self.log_to_mlflow:
            return

        checkpoint = int(self.num_timesteps)
        env_idx = int(locals_["i"])

        # Durante una evaluation num_timesteps resta fisso:
        # identifica il checkpoint del modello valutato.
        if checkpoint != self._artifact_checkpoint:
            if any(self._artifact_episode_steps.values()):
                raise RuntimeError("Cambio di checkpoint con dati di evaluation non chiusi.")

            self._artifact_checkpoint = checkpoint
            self._artifact_episode_ids.clear()
            self._artifact_episode_steps.clear()

        payload = locals_["info"].get("eval_artifacts")
        if payload is None:
            raise RuntimeError(
                "Dati eval_artifacts mancanti: controllare la configurazione "
                "dell'ambiente di evaluation."
            )

        steps = self._artifact_episode_steps.setdefault(env_idx, [])
        steps.append(payload)

        # Lo step terminale viene aggiunto PRIMA del salvataggio.
        if not bool(locals_["done"]):
            return

        episode_id = self._artifact_episode_ids.get(env_idx, 0)

        snapshot_dates = self.eval_env.get_attr(
            "artifact_snapshot_dates",
            indices=env_idx,
        )[0]

        artifact = {
            "schema_version": 3,
            "decision_end_inclusive": True,
            "checkpoint_timesteps": checkpoint,
            "env_index": env_idx,
            "episode_id": episode_id,
            "snapshot_dates": sorted(value.isoformat() for value in snapshot_dates),
            "steps": steps,
        }

        mlflow.log_dict(
            artifact,
            (
                f"eval_data/{checkpoint:08d}_timesteps/"
                f"env_{env_idx:02d}/episode_{episode_id:05d}.json"
            ),
        )

        log_eval_artifact_plots(artifact)

        self._artifact_episode_steps[env_idx] = []
        self._artifact_episode_ids[env_idx] = episode_id + 1
