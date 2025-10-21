from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.hotel.room import RoomCategory
from pricing_advertising.plots.plot import plot_and_log_figures
from pricing_advertising.requests.request import Request
from pricing_advertising.utils.constants import NUM_COEFFICIENTS


class PlotCallback(BaseCallback):
    def __init__(self, log_to_mlflow: bool = True, occ_hist_samples: int = 40) -> None:
        super().__init__()
        self.n_envs: int | None = None  # type hint per Pylance
        self.rows: list[dict[str, Any]] = []
        self.log_to_mlflow = log_to_mlflow
        self.occ_hist_samples = occ_hist_samples
        self._num_rooms_cache: dict[int, list[int]] = {}

    # ========== HELPER FUNCTIONS FOR SubprocVecEnv ==========

    # Returns the “name” attribute of the env index idx
    def _get_attr(self, name: str, idx: int):
        return self.training_env.get_attr(name, indices=idx)[0]

    # Calls a method (defined in env) in the subprocess
    def _env_method(self, method_name: str, *args, idx: int, **kwargs) -> Any:  # noqa: ANN003, ANN401
        return self.training_env.env_method(method_name, *args, indices=idx, **kwargs)[0]

    def _append_row(
        self, env_idx: int, day_idx: int, category: str, kind: str, value: float
    ) -> None:
        self.rows.append(
            {
                "date": int(day_idx),
                "category": category,
                "kind": kind,
                "value": float(value),
                "env": int(env_idx),
            }
        )

    # Returns [rooms_cat1, rooms_cat2, rooms_cat3] for env idx, with caching
    def _get_num_rooms_by_cat(self, idx: int) -> list[int]:
        if idx in self._num_rooms_cache:
            return self._num_rooms_cache[idx]

        num_cats: int = self._get_attr("num_room_categories", idx)
        out: list[int] = []
        for c in range(1, num_cats + 1):
            n = self._env_method("get_num_rooms_per_category", RoomCategory(c), idx=idx)
            out.append(int(n))

        self._num_rooms_cache[idx] = out
        return out

    # ========== UTILS FUNCTIONS ==========

    # 1) INITIAL REQUESTS
    def _log_initial_requests_from_generator(self, env_idx: int) -> None:
        gen = self._get_attr("request_generator", env_idx)
        start_sim = self._get_attr("start_simulation", env_idx)
        end_sim = self._get_attr("end_simulation", env_idx)

        if gen is None or start_sim is None or end_sim is None:
            return

        counts = defaultdict(int)
        current = start_sim
        while current < end_sim:
            reqs = gen.requests_by_day.get(current, [])
            for r in reqs:
                cat = r.category  # <-- usa la proprietà già definita in Request
                idx = cat.value if hasattr(cat, "value") else int(cat)
                counts[idx - 1] += 1
            current += timedelta(days=1)

        # Log nel dataframe per plot.py
        for c_idx, total in sorted(counts.items()):
            self._append_row(env_idx, 0, str(c_idx + 1), "initial_requests", float(total))

    # 2) OCCUPANCY DISTRIBUTION
    def _log_occupancy_dist_sample(
        self, env_idx: int, current_day_index: int, num_rooms: list[int]
    ) -> None:
        # 1) Read from the env the info you need
        simulation_days = self._get_attr("total_days", env_idx)
        occ = self._get_attr("occupancy", env_idx)  # shape: (num_cat, extended_days)
        # use only the real horizon
        occ_h = occ[:, :simulation_days]  # shape: (num_cat, simulation_days)

        C, D = occ_h.shape

        # 3) Sample uniformly across the horizon (category, day)
        total = C * D
        sample_size = min(self.occ_hist_samples, total)
        rng = np.random.default_rng()
        idx_flat = rng.choice(total, size=sample_size, replace=False)
        cats = idx_flat // D
        days = idx_flat % D

        # 4) Normalise with the number of rooms in the correct category
        for c, d in zip(cats, days, strict=True):
            denom = max(1, num_rooms[c])
            v = float(occ_h[c, d] / denom)
            self._append_row(
                env_idx,
                int(current_day_index),
                str(c + 1),
                "occupancy_dist_sample",
                v,
            )

    # 3) PRICES

    # 4) ACTION CURVE
    def _log_action_curve(
        self, env_idx: int, current_day_index: int, num_cats: int, actions_env: np.ndarray
    ) -> None:
        for cat_idx in range(num_cats):
            try:
                action_val = float(np.mean(actions_env[cat_idx]))
            except Exception:
                action_val = float(actions_env)
            self._append_row(
                env_idx, current_day_index, f"Cat{cat_idx + 1}", "action_curve", action_val
            )

    def _probe_price_for_advance(
        self, idx: int, advance: int, category: RoomCategory, actions: np.ndarray
    ) -> float | None:
        today = self._get_attr("today", idx)
        extended_end_simulation = self._get_attr("extended_end_simulation", idx)

        res_date = today + timedelta(days=advance)
        if res_date >= extended_end_simulation:
            return None

        # richiesta fittizia (1 notte) coerente con la tua Request
        req = Request(
            t_request=today,
            t_reservation=res_date,
            days_stay=1,
            num_guests=category.value,
        )

        # chiamiamo il metodo dell'env nel subprocesso (ritorna list[list[float]])
        actions = np.asarray(actions, dtype=np.float64)
        num_room_categories = self._get_attr("num_room_categories", idx)
        actions = actions.reshape(num_room_categories, NUM_COEFFICIENTS)
        actions = np.clip(actions, 0.0, 1.0)  # <<< importante
        prices_all = self._env_method("_generate_prices", np.array([req]), actions, idx=idx)
        if not prices_all or not prices_all[0]:
            return None
        return float(np.mean(prices_all[0]))

    # 5) OCCUPANCY PERCENTAGE AT EACH DAY
    def _log_occupancy_pct(
        self, env_idx: int, current_day_index: int, num_cats: int, num_rooms: list[int]
    ) -> None:
        occupancy = self._get_attr("occupancy", env_idx)
        for cat_idx in range(num_cats):
            occ_raw = float(occupancy[cat_idx, current_day_index])
            percentage = occ_raw / max(1, num_rooms[cat_idx]) * 100.0
            self._append_row(
                env_idx, current_day_index, f"Cat{cat_idx + 1}", "occupancy_pct", percentage
            )

    def _log_probe_prices(
        self, env_idx: int, current_day_index: int, num_cats: int, actions_env: np.ndarray
    ) -> None:
        for k in (1, 15, 60):
            for cat_val in range(1, num_cats + 1):
                prob = self._probe_price_for_advance(env_idx, k, RoomCategory(cat_val), actions_env)
                if prob is not None:
                    self._append_row(
                        env_idx, current_day_index, f"Cat{cat_val}", f"advance_{k}", prob
                    )

    # ========== CALLBACK SB3 ==========

    # ========== On Step ==========
    def _on_step(self) -> bool:
        n_envs = self.training_env.num_envs
        # 1) Normnalize actions
        raw_actions = self.locals.get("actions")
        actions_np = None
        if raw_actions is not None:
            try:
                actions_np = np.asarray(raw_actions, dtype=float)
            except Exception:
                actions_np = None

        for env_idx in range(n_envs):
            today = self._get_attr("today", env_idx)
            start_sim = self._get_attr("start_simulation", env_idx)
            current_day_index = (today - start_sim).days

            num_cats = self._get_attr("num_room_categories", env_idx)
            num_rooms = self._get_num_rooms_by_cat(env_idx)
            # 2) Extract the actions of the current env (if available)
            actions_env = None
            if actions_np is not None:
                if actions_np.ndim >= 1 and actions_np.shape[0] == n_envs:
                    actions_env = actions_np[env_idx]
                else:
                    actions_env = actions_np

            # 3) Log metrics
            if actions_env is not None:
                self._log_action_curve(env_idx, current_day_index, num_cats, actions_env)
                self._log_probe_prices(env_idx, current_day_index, num_cats, actions_env)
            self._log_occupancy_pct(env_idx, current_day_index, num_cats, num_rooms)
            self._log_occupancy_dist_sample(env_idx, current_day_index, num_rooms)

        return True

    # ========== On Training ==========
    def _on_training_start(self) -> None:
        if self.n_envs is None:
            self.n_envs = getattr(self.training_env, "num_envs", 1)
        n_envs = self.n_envs if self.n_envs is not None else 1
        for env_idx in range(n_envs):
            self._log_initial_requests_from_generator(env_idx)

    def on_training_end(self) -> None:
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)

        # if you want to aggregate all env in a single chart, that's fine
        # if you want to separate by env, you can filter by the env column
        csv_text = df.to_csv(index=False)
        if self.log_to_mlflow:
            mlflow.log_text(csv_text, "episode_log.csv")

        # Generates and logs images
        plot_and_log_figures(df, log_to_mlflow=self.log_to_mlflow)
