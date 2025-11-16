from __future__ import annotations

from datetime import date, timedelta
from typing import Any, cast

import mlflow
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.plots.plot import plot_and_log_figures
from pricing_advertising.pricing_model.bar_table import get_price, n_levels
from pricing_advertising.requests.request_generator import RequestGenerator
from pricing_advertising.utils.constants import HIGH_SEASON, NUM_COEFFICIENTS


class PlotCallback(BaseCallback):
    def __init__(self, log_to_mlflow: bool = True) -> None:
        super().__init__()
        self.n_envs: int | None = None  # type hint per Pylance
        self.rows: list[dict[str, Any]] = []
        self.log_to_mlflow = log_to_mlflow
        self._num_rooms_cache: dict[int, list[int]] = {}
        # Store the pairs (category, day) already logged for each env
        self._occ_seen: dict[int, set[tuple[int, int]]] = {}
        self._initial_requests_logged = False

    # ========== HELPER FUNCTIONS FOR SubprocVecEnv ==========

    # Returns the “name” attribute of the env index idx
    def _get_attr(self, name: str, idx: int):
        return self.training_env.get_attr(name, indices=idx)[0]

    # Calls a method (defined in env) in the subprocess
    def _env_method(self, method_name: str, *args, idx: int, **kwargs) -> Any:  # noqa: ANN003, ANN401
        return self.training_env.env_method(method_name, *args, indices=idx, **kwargs)[0]

    def _append_row(
        self, env_idx: int, day_idx: int, category: str, kind: str, value: float, season: str | None
    ) -> None:
        self.rows.append(
            {
                "day": int(day_idx),
                "category": category,
                "kind": kind,
                "value": float(value),
                "env": int(env_idx),
                "season": str(season),
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

    def _season_from_date(self, dt: date) -> str:
        m = dt.month
        if m in HIGH_SEASON:
            return "High"
        else:
            return "Low"

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

        # If gen has the method we call it in order to ensure that requests_by_day is not empty
        if (
            gen is not None
            and hasattr(gen, "ensure_precomputed")
            and callable(gen.ensure_precomputed)
        ):
            gen.ensure_precomputed()

        # Counter for each category
        totals_by_cat: dict[RoomCategory, int] = {cat: 0 for cat in RoomCategory}
        current = start_sim
        # Loop in the simulation period
        while current < end_sim:
            # Take the requests generated for that day
            reqs = gen.requests_by_day.get(current, [])
            day_idx = (current - start_sim).days
            total_today = len(reqs)
            season = self._season_from_date(current)
            self._append_row(
                env_idx=env_idx,
                day_idx=day_idx,
                category="all",
                kind="requests_daily_total",
                value=int(total_today),
                season=season,
            )

            # For each day
            per_cat_today: dict[int, int] = {}
            for r in reqs:
                cat_val = r.category.value
                per_cat_today[cat_val] = per_cat_today.get(cat_val, 0) + 1

            for cval, cnt in per_cat_today.items():
                self._append_row(
                    env_idx=env_idx,
                    day_idx=day_idx,
                    category=str(cval),
                    kind="requests_daily_cat",
                    value=float(cnt),
                    season=season,
                )

            for r in reqs:
                totals_by_cat[r.category] += 1
            current += timedelta(days=1)

        total_requests = sum(totals_by_cat.values())
        print(f"[Env {env_idx}] Total number of requests: {total_requests}")

        # Sort the counters by key (category)
        for cat in sorted(totals_by_cat.keys(), key=lambda c: c.value):
            # Number of total requests for that category
            total = totals_by_cat[cat]
            self._append_row(
                env_idx=env_idx,
                day_idx=0,
                category=str(cat.value),
                kind="initial_requests",
                value=float(total),
                season=None,
            )

    # 2) OCCUPANCY DISTRIBUTION
    def _log_occupancy_dist(self, env_idx: int) -> None:
        simulation_days = self._get_attr("total_days", env_idx)
        if simulation_days is None:
            return

        # prendi direttamente la matrice occupancy (shape: num_cat × extended_days)
        occ = self._get_attr("occupancy", env_idx)
        if occ is None:
            return

        occ_h = occ[:, :simulation_days]  # usa solo i giorni reali
        C, D = occ_h.shape
        num_rooms = self._get_num_rooms_by_cat(env_idx)

        # controlla se i valori sono già frazioni 0–1
        mx = float(np.nanmax(occ_h)) if occ_h.size else 0.0
        already_fraction = mx <= 1.0 + 1e-6

        for c in range(C):
            denom = max(1, int(num_rooms[c]))
            for d in range(D):
                raw = float(occ_h[c, d])

                # se già normalizzato → usa così, altrimenti normalizza
                v = np.clip(raw, 0.0, 1.0) if already_fraction else np.clip(raw / denom, 0.0, 1.0)

                self._append_row(
                    env_idx=env_idx,
                    day_idx=int(d),
                    category=str(c + 1),
                    kind="occupancy_dist",
                    value=float(v),
                    season=None,
                )

    # OCCUPANCY PERCENTAGE AT EACH DAY
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
        """
        Logga i conteggi Accepted/Rejected (per categoria + totale) a fine training.
        Richiede idealmente che l'env esponga:
        - accepted_by_day: np.ndarray (C, extended_days) di interi
        - rejected_by_day: np.ndarray (C, extended_days) di interi
        Fallback: prova attributi cumulativi se le matrici non esistono.
        """
        simulation_days = self._get_attr("total_days", env_idx)
        if simulation_days is None:
            return

        # Prova matrici per giorno (preferite)
        acc = None
        rej = None
        try:
            acc = self._get_attr("accepted_by_day", env_idx)
            rej = self._get_attr("rejected_by_day", env_idx)
        except Exception:
            acc = None
            rej = None

        if acc is not None and rej is not None:
            acc_h = acc[:, :simulation_days]  # shape (C, D)
            rej_h = rej[:, :simulation_days]
            C, D = acc_h.shape

            # Per categoria
            for c in range(C):
                accepted_c = int(acc_h[c].sum())
                rejected_c = int(rej_h[c].sum())
                self._append_row(
                    env_idx,
                    day_idx=D - 1,
                    category=str(c + 1),
                    kind="accepted_count",
                    value=float(accepted_c),
                    season=None,
                )
                self._append_row(
                    env_idx,
                    day_idx=D - 1,
                    category=str(c + 1),
                    kind="rejected_count",
                    value=float(rejected_c),
                    season=None,
                )

            # Totale "all"
            accepted_all = int(acc_h.sum())
            rejected_all = int(rej_h.sum())
            self._append_row(
                env_idx,
                day_idx=D - 1,
                category="all",
                kind="accepted_count",
                value=float(accepted_all),
                season=None,
            )
            self._append_row(
                env_idx,
                day_idx=D - 1,
                category="all",
                kind="rejected_count",
                value=float(rejected_all),
                season=None,
            )
            return

        # ---- Fallback
        try:
            accepted_all = int(self._get_attr("accepted_total", env_idx))
            rejected_all = int(self._get_attr("rejected_total", env_idx))
            D = int(simulation_days)
            self._append_row(
                env_idx,
                day_idx=D - 1,
                category="all",
                kind="accepted_count",
                value=float(accepted_all),
                season=None,
            )
            self._append_row(
                env_idx,
                day_idx=D - 1,
                category="all",
                kind="rejected_count",
                value=float(rejected_all),
                season=None,
            )
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
                env_idx,
                current_day_index,
                f"Cat{cat_idx + 1}",
                "action_curve",
                action_val,
                season=None,
            )
        # Generic curve: mean of all category
        try:
            overall = float(np.mean(actions_env))
        except Exception:
            overall = float(np.mean(vals))
        self._append_row(
            env_idx=env_idx,
            day_idx=current_day_index,
            category="All",
            kind="action_curve",
            value=overall,
            season=None,
        )

    # 5) PRICE CURVE
    def _log_price_curve(self, env_idx: int) -> None:
        total_days = self._get_attr("total_days", env_idx)
        bar_levels = self._get_attr("bar_levels", env_idx)
        num_cats = self._get_attr("num_room_categories", env_idx)
        if total_days is None or bar_levels is None or num_cats is None:
            return

        max_level = n_levels() - 1

        # use the real horizon
        for c in range(int(num_cats)):
            category_enum = RoomCategory(c + 1)
            for d in range(int(total_days)):
                level = int(bar_levels[c, d])
                level = max(0, min(level, max_level))
                price = float(get_price(category_enum.name, level))
                self._append_row(env_idx, d, f"Cat{c + 1}", "price_curve", price, season=None)

    # 6) BAR LEVEL CURVE
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
                level = int(bar_levels[c, d])
                self._append_row(
                    env_idx=env_idx,
                    day_idx=d,
                    category=f"Cat{c + 1}",
                    kind="bar_level",
                    value=float(level),
                    season=None,
                )

    # PRICES FOR ADVANCES
    def _log_probe_prices(
        self, env_idx: int, current_day_index: int, num_cats: int, actions_env: np.ndarray
    ) -> None:
        """
        Logga il prezzo 'quote' per una notte alla data today+advance, per ogni categoria.
        NON modifica lo stato dell'env (niente update dei BAR!).
        """
        for k in (1, 15, 60):
            for cat_val in range(1, num_cats + 1):
                p = self._probe_price_for_advance(env_idx, k, RoomCategory(cat_val))
                if p is not None:
                    self._append_row(
                        env_idx, current_day_index, f"Cat{cat_val}", f"advance_{k}", p, season=None
                    )

    def _probe_price_for_advance(
        self, idx: int, advance: int, category: RoomCategory
    ) -> float | None:
        """
        Restituisce il prezzo (1 notte) alla data today+advance per 'category',
        calcolato dal BAR corrente, SENZA aggiornare livelli/bar o altro.
        """
        today = self._get_attr("today", idx)
        start_sim = self._get_attr("start_simulation", idx)
        extended_end_simulation = self._get_attr("extended_end_simulation", idx)
        bar_levels = self._get_attr("bar_levels", idx)

        # calcolo l’indice del giorno di prenotazione (reservation day)
        res_date = today + timedelta(days=advance)
        if res_date >= extended_end_simulation:
            return None

        day_idx = (res_date - start_sim).days
        if day_idx < 0:
            return None

        cat_idx = category.value - 1
        # prendo il livello BAR attuale
        level = int(bar_levels[cat_idx, day_idx])

        # mappo sul prezzo dalla tabella BAR
        price = get_price(category.name, level)
        return float(price)

    # BERS. COEFFICIENTS
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

        # Clipping at bounds of action space: to guarantee that the coefficients remain between 0 and 1
        low = np.asarray(self.training_env.get_attr("action_space")[0].low, dtype=float).reshape(
            num_cats, NUM_COEFFICIENTS
        )
        high = np.asarray(self.training_env.get_attr("action_space")[0].high, dtype=float).reshape(
            num_cats, NUM_COEFFICIENTS
        )
        a = np.clip(a, low, high)

        # Log
        for c in range(num_cats):
            for k in range(NUM_COEFFICIENTS):
                self._append_row(
                    env_idx,
                    current_day_index,
                    f"Cat{c + 1}_k{k}",
                    "action_coeff",
                    float(a[c, k]),
                    season=None,
                )

    # ========== CALLBACK SB3 ==========

    # ========== ON STEP ==========
    def _on_step(self) -> bool:
        n_envs = self.training_env.num_envs
        # 1) Normnalize actions
        raw_actions = self.locals.get("actions")
        actions_np = None
        if raw_actions is not None:
            raw = np.asarray(raw_actions, dtype=float)
            # prendi gli spazi azione dei sotto-env (lista lunga n_envs)
            action_spaces = self.training_env.get_attr("action_space")
            # usa il primo env (indice 0)
            low = np.asarray(action_spaces[0].low, dtype=float)
            high = np.asarray(action_spaces[0].high, dtype=float)

            # riscalatura: [-1,1]  ->  [low, high]
            env_actions = low + (raw + 1.0) * 0.5 * (high - low)

            print("min/max raw:", raw.min(), raw.max())
            print("min/max env:", env_actions.min(), env_actions.max())
            print("action_space low/high:", low[:5], high[:5])
            try:
                actions_np = np.asarray(env_actions, dtype=float)
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
                # shape: (n_envs, num_cats, NUM_COEFFICIENTS)
                actions_env = actions_np[env_idx] if actions_np.shape[0] == n_envs else actions_np

            # 3) Log metrics
            if actions_env is not None:
                self._log_action_curve(env_idx, current_day_index, num_cats, actions_env)
                self._log_probe_prices(env_idx, current_day_index, num_cats, actions_env)
                self._log_action_coeffs(env_idx, current_day_index, actions_env)
            self._log_occupancy_pct(env_idx, current_day_index, num_cats, num_rooms)

        return True

    # ========== ON TRAINING START ==========
    def _on_training_start(self) -> None:
        if self.n_envs is None:
            self.n_envs = getattr(self.training_env, "num_envs", 1)
        # if you have already logged the requests in the previous cycle, don't do aganin!
        if self._initial_requests_logged:
            return
        n_envs = self.n_envs if self.n_envs is not None else 1
        for env_idx in range(n_envs):
            self._log_initial_requests_from_generator(env_idx)
        self._initial_requests_logged = True

    # ========== ON TRAINING END ==========
    def on_training_end(self) -> None:
        n_envs = self.training_env.num_envs

        # 1) Add the occupancy raws for each env
        for env_idx in range(n_envs):
            self._log_occupancy_dist(env_idx)
            self._log_acceptance_counts(env_idx)
            self._log_price_curve(env_idx)
            self._log_bar_levels(env_idx)

        # 2) Create the dataframe
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)

        # 3) Save the CSV
        if self.log_to_mlflow:
            mlflow.log_text(df.to_csv(index=False), "episode_log.csv")

        # 4) Generates plots
        plot_and_log_figures(df, log_to_mlflow=self.log_to_mlflow)
