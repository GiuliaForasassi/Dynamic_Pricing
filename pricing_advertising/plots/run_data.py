"""Compact per-run data that the parent run of a multi-seed experiment averages."""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.artifacts
from mlflow.exceptions import MlflowException

DATA_DIR = "aggregation_data"


def log_run_data(name: str, payload: dict[str, Any]) -> None:
    mlflow.log_dict(payload, f"{DATA_DIR}/{name}.json")


def load_run_data(run_id: str, name: str) -> dict[str, Any] | None:
    try:
        return mlflow.artifacts.load_dict(f"runs:/{run_id}/{DATA_DIR}/{name}.json")
    except (MlflowException, OSError):
        return None
