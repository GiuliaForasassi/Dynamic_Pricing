from __future__ import annotations

from datetime import date, timedelta
from time import perf_counter
from typing import Any, cast

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.plots.plot import plot_and_log_figures
from pricing_advertising.pricing_model.bar_table import get_price, n_levels
from pricing_advertising.requests.request_generator import RequestGenerator
from pricing_advertising.utils.constants import HIGH_SEASON, NUM_COEFFICIENTS


class PlotCallback(BaseCallback):
    def __init__(
        self,
        log_to_mlflow: bool = True,
        track_env_idx: int = 0,  # default is env 0
        plot_on_episode_end: bool = True,
        log_initial_requests_once: bool = True,
    ) -> None:
        super().__init__()
        self.log_to_mlflow = log_to_mlflow
        self.track_env_idx = int(track_env_idx)
        self.plot_on_episode_end = bool(plot_on_episode_end)
        self.log_initial_requests_once = bool(log_initial_requests_once)

        # buffer of the current episode
        self._ep_rows: list[dict[str, Any]] = []
        self._episode_id: int = 0

        # caching
        self._num_rooms_cache: dict[int, list[int]] = {}
        self._initial_requests_logged: bool = False

    # ========== HELPER FUNCTIONS FOR SubprocVecEnv ==========

    # Returns the “name” attribute of the env index idx
    def _get_attr(self, name: str, idx: int):
        return self.training_env.get_attr(name, indices=idx)[0]

    # Calls a method (defined in env) in the subprocess
    def _env_method(self, method_name: str, *args, idx: int, **kwargs) -> Any:  # noqa: ANN401
        return self.training_env.env_method(method_name, *args, indices=idx, **kwargs)[0]

    def _append_row(
        self,
        env_idx: int,
        day_idx: int,
        category: str,
        kind: str,
        value: float,
        season: str | None,
    ) -> None:
        # IMPORTANT: non fare str(None) -> "None"
        row = {
            "episode": int(self._episode_id),
            "day": int(day_idx),
            "category": category,
            "kind": kind,
            "value": float(value),
            "env": int(env_idx),
            "season": season,  # può restare None
        }
        self._ep_rows.append(row)

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

    def _season_from_date(self, dt: date) -> str:
        return "High" if dt.month in HIGH_SEASON else "Low"

    # ========== UTILS FUNCTIONS ==========

    # 1) INITIAL REQUESTS
    def _log_initial_requests_from_generator(self, env_idx: int) -> None:
        # Take from the env the request_generator
        gen = cast(RequestGenerator | None, self._get_attr("request_generator", env_idx))
        # Take from the env the dates of start and end simulation
        start_sim = self._get_attr("start_simulation", env_idx)
        end_sim = self._get_attr("end_simulation", env_idx)
        if gen is None or start_sim is None or end_sim is None:
            return

        # ensure precomputed
        if hasattr(gen, "ensure_precomputed") and callable(gen.ensure_precomputed):
            gen.ensure_precomputed()

        # Counter for each category
        totals_by_cat: dict[RoomCategory, int] = {cat: 0 for cat in RoomCategory}
        current = start_sim

        # Loop in the simulation period
        while current < end_sim:
            # Take the requests generated for that day
            reqs = gen.requests_by_day.get(current, [])
            day_idx = (current - start_sim).days
            season = self._season_from_date(current)

            self._append_row(
                env_idx=env_idx,
                day_idx=day_idx,
                category="all",
                kind="requests_daily_total",
                value=float(len(reqs)),
                season=season,
            )

            # For each day:
            per_cat_today: dict[int, int] = {}
            for r in reqs:
                cat_val = r.category.value
                per_cat_today[cat_val] = per_cat_today.get(cat_val, 0) + 1
                totals_by_cat[r.category] += 1

            for cval, cnt in per_cat_today.items():
                self._append_row(
                    env_idx=env_idx,
                    day_idx=day_idx,
                    category=str(cval),
                    kind="requests_daily_cat",
                    value=float(cnt),
                    season=season,
                )

            current += timedelta(days=1)

        # Sort the counters by key (category)
        for cat in sorted(totals_by_cat.keys(), key=lambda c: c.value):
            # Add the number of total requests for that category
            self._append_row(
                env_idx=env_idx,
                day_idx=0,
                category=str(cat.value),
                kind="initial_requests",
                value=float(totals_by_cat[cat]),
                season=None,
            )

    # 2) OCCUPANCY
    # Occupancy distribution
    def _log_occupancy_dist(self, env_idx: int) -> None:
        simulation_days = self._get_attr("total_days", env_idx)
        if simulation_days is None:
            return

        # prendi direttamente la matrice occupancy (shape: num_cat x extended_days)
        occ = self._get_attr("occupancy", env_idx)
        if occ is None:
            return

        occ_h = occ[:, :simulation_days]  # usa solo i giorni reali
        C, D = occ_h.shape
        num_rooms = self._get_num_rooms_by_cat(env_idx)

        for c in range(C):
            denom = max(1, int(num_rooms[c]))
            for d in range(D):
                raw = float(occ_h[c, d])

                # se già normalizzato → usa così, altrimenti normalizza
                v = np.clip(raw / denom, 0.0, 1.0)

                self._append_row(
                    env_idx=env_idx,
                    day_idx=int(d),
                    category=str(c + 1),
                    kind="occupancy_dist",
                    value=float(v),
                    season=None,
                )

    # Occupancy percentage at each day
    def _log_occupancy_pct(
        self, env_idx: int, current_day_index: int, num_cats: int, num_rooms: list[int]
    ) -> None:
        occupancy = self._get_attr("occupancy", env_idx)
        for cat_idx in range(num_cats):
            occ_raw = float(occupancy[cat_idx, current_day_index])
            percentage = occ_raw / max(1, num_rooms[cat_idx]) * 100.0
            self._append_row(
                env_idx,
                current_day_index,
                f"Cat{cat_idx + 1}",
                "occupancy_pct",
                percentage,
                season=None,
            )

    # 3) ACCEPTANCE
    def _log_acceptance_counts(self, env_idx: int) -> None:
        simulation_days = self._get_attr("total_days", env_idx)
        if simulation_days is None:
            return

        acc = None
        rej = None
        try:
            acc = self._get_attr("accepted_by_day", env_idx)
            rej = self._get_attr("rejected_by_day", env_idx)
        except Exception:
            acc = None
            rej = None

        D = int(simulation_days)

        if acc is not None and rej is not None:
            acc_h = acc[:, :D]
            rej_h = rej[:, :D]
            C, _ = acc_h.shape

            # For each category
            for c in range(C):
                self._append_row(
                    env_idx, D - 1, str(c + 1), "accepted_count", float(acc_h[c].sum()), None
                )
                self._append_row(
                    env_idx, D - 1, str(c + 1), "rejected_count", float(rej_h[c].sum()), None
                )

            # Total
            self._append_row(env_idx, D - 1, "all", "accepted_count", float(acc_h.sum()), None)
            self._append_row(env_idx, D - 1, "all", "rejected_count", float(rej_h.sum()), None)
            return

        # fallback
        try:
            accepted_all = int(self._get_attr("accepted_total", env_idx))
            rejected_all = int(self._get_attr("rejected_total", env_idx))
            self._append_row(env_idx, D - 1, "all", "accepted_count", float(accepted_all), None)
            self._append_row(env_idx, D - 1, "all", "rejected_count", float(rejected_all), None)
        except Exception:
            return

    # 4) ACTION CURVE
    def _log_action_curve(
        self, env_idx: int, current_day_index: int, num_cats: int, actions_env: np.ndarray
    ) -> None:
        vals: list[float] = []
        for cat_idx in range(num_cats):
            try:
                action_val = float(np.mean(actions_env[cat_idx]))
            except Exception:
                action_val = float(actions_env)
            vals.append(action_val)
            self._append_row(
                env_idx, current_day_index, f"Cat{cat_idx + 1}", "action_curve", action_val, None
            )

        # Generic curve: mean of all category
        try:
            overall = float(np.mean(actions_env))
        except Exception:
            overall = float(np.mean(vals))
        self._append_row(env_idx, current_day_index, "all", "action_curve", overall, None)

    # 5) BERST. COEFFICIENTS
    def _log_action_coeffs(
        self, env_idx: int, current_day_index: int, actions_env: np.ndarray
    ) -> None:
        num_cats = int(self._get_attr("num_room_categories", env_idx))
        try:
            a = np.asarray(actions_env, dtype=float).reshape(num_cats, NUM_COEFFICIENTS)
        except Exception:
            # For security: if for some reason actions_env doesn't have the expected form (ex: is a 1D vector)
            # it is duplicated for all 3 categories
            vec = np.asarray(actions_env, dtype=float).ravel()
            a = np.tile(vec, (num_cats, 1))[:, :NUM_COEFFICIENTS]

        action_spaces = self.training_env.get_attr("action_space")
        # Clipping at bounds of action space: to guarantee that the coefficients remain between 0 and 1
        low = np.asarray(action_spaces[0].low, dtype=float).reshape(num_cats, NUM_COEFFICIENTS)
        high = np.asarray(action_spaces[0].high, dtype=float).reshape(num_cats, NUM_COEFFICIENTS)
        a = np.clip(a, low, high)

        for c in range(num_cats):
            for k in range(NUM_COEFFICIENTS):
                self._append_row(
                    env_idx,
                    current_day_index,
                    f"Cat{c + 1}_k{k}",
                    "action_coeff",
                    float(a[c, k]),
                    None,
                )

    # 6) PRICE CURVE
    def _log_price_curve(self, env_idx: int) -> None:
        total_days = self._get_attr("total_days", env_idx)
        bar_levels = self._get_attr("bar_levels", env_idx)
        num_cats = self._get_attr("num_room_categories", env_idx)
        if total_days is None or bar_levels is None or num_cats is None:
            return

        max_level = n_levels() - 1
        D = int(total_days)

        # use the real horizon
        for c in range(int(num_cats)):
            category_enum = RoomCategory(c + 1)
            for d in range(D):
                level = int(bar_levels[c, d])
                level = max(0, min(level, max_level))
                price = float(get_price(category_enum.name, level))
                self._append_row(env_idx, d, f"Cat{c + 1}", "price_curve", price, None)

    # 7) BAR LEVEL CURVE
    def _log_bar_levels(self, env_idx: int) -> None:
        total_days = self._get_attr("total_days", env_idx)
        bar_levels = self._get_attr("bar_levels", env_idx)
        num_cats = self._get_attr("num_room_categories", env_idx)
        if total_days is None or bar_levels is None or num_cats is None:
            return

        D = int(total_days)
        C = int(num_cats)
        for c in range(C):
            for d in range(D):
                self._append_row(
                    env_idx, d, f"Cat{c + 1}", "bar_level", float(bar_levels[c, d]), None
                )

    # 8) REWARD
    def _log_reward_daily(self, env_idx: int, current_day_index: int, reward_val: float) -> None:
        self._append_row(env_idx, current_day_index, "all", "reward_daily", float(reward_val), None)

    # ========== LOG EPISODE ==========
    def _finalize_and_log_episode(self) -> None:
        # Check if there are data to save
        if not self._ep_rows:
            return

        # Converts the buffer of episodes into a dataframe
        df = pd.DataFrame(self._ep_rows)

        # Save CSV of the episode into MLflow
        if self.log_to_mlflow:
            mlflow.log_text(df.to_csv(index=False), f"episodes/episode_{self._episode_id:05d}.csv")

        # Generates and log graphs (prefix used in plot in order to not overwrite)
        if self.plot_on_episode_end:
            plot_and_log_figures(
                df,
                log_to_mlflow=self.log_to_mlflow,
                prefix=f"episode_{self._episode_id:05d}",
            )

        # Reset buffer
        self._ep_rows.clear()
        # Go to the next episode
        self._episode_id += 1

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

        num_cats = int(self._get_attr("num_room_categories", env_idx))
        num_rooms = self._get_num_rooms_by_cat(env_idx)

        # ---- rewards ----
        # self.locals is a dict that SB3 pass to the call back and it contains: rewards, dones, infos, actions...
        raw_rewards = self.locals.get("rewards")
        if raw_rewards is not None:
            r = float(np.asarray(raw_rewards, dtype=float).ravel()[env_idx])
        self._log_reward_daily(env_idx, current_day_index, r)

        # ---- actions ----
        raw_actions = self.locals.get("actions")
        if raw_actions is not None:
            raw = np.asarray(raw_actions, dtype=float)

            action_spaces = self.training_env.get_attr("action_space")
            low = np.asarray(action_spaces[0].low, dtype=float)
            high = np.asarray(action_spaces[0].high, dtype=float)

            # riscalatura: [-1,1]  ->  [low, high]
            env_actions = low + (raw + 1.0) * 0.5 * (high - low)
            actions_np = np.asarray(env_actions, dtype=float)

            actions_env = actions_np[env_idx]
            self._log_action_curve(env_idx, current_day_index, num_cats, actions_env)
            self._log_action_coeffs(env_idx, current_day_index, actions_env)

        # ---- occupancy pct (ogni step) ----
        self._log_occupancy_pct(env_idx, current_day_index, num_cats, num_rooms)

        # ---- end episode detection ----
        dones = self.locals.get("dones")
        if dones is not None and bool(np.asarray(dones, dtype=bool).ravel()[env_idx]):
            # log metrics of final episode
            self._log_acceptance_counts(env_idx)
            self._log_price_curve(env_idx)
            self._log_bar_levels(env_idx)
            self._log_occupancy_dist(env_idx)

            # flush: df + csv + figure, and then reset
            self._finalize_and_log_episode()

        return True

    # ========== ON TRAINING ==========
    def _on_training_start(self) -> None:
        if self._initial_requests_logged:
            return

        n_envs = int(getattr(self.training_env, "num_envs", 1))
        env_idx = self.track_env_idx

        if env_idx < 0 or env_idx >= n_envs:
            return

        if self.log_initial_requests_once:
            self._log_initial_requests_from_generator(env_idx)
            self._initial_requests_logged = True

    def _on_training_end(self) -> None:
        # ep_rows raccoglioe i 365 giorni dell'episodio corrent (solo env tracciato)
        if self._ep_rows:
            self._finalize_and_log_episode()
