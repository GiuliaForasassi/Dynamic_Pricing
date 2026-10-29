import os

import mlflow
import pytest


@pytest.fixture(autouse=True)
def isolated_mlflow_tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield
    # Runs auto-created by mlflow.log_* stay active globally and would point to a deleted database.
    while mlflow.active_run():
        mlflow.end_run()
    # set_experiment keeps the experiment id of the deleted database.
    os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
    mlflow.tracking.fluent._active_experiment_id = None
