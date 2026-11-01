import cProfile
import io
import os
import pstats
import re
from datetime import date
from pathlib import Path

from pricing_advertising.pricing_model.bar_table import set_bar_table_from_csv
from pricing_advertising.training.ppo_training import run_simulation
from pricing_advertising.utils.constants import EVAL_SEEDS, SEEDS


def main() -> None:
    # Same backend as `make mlflow-ui`; the ./mlruns file store is no longer accepted by MLflow.
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    # Load bar table
    csv_path = Path("data") / "bar_table.csv"
    set_bar_table_from_csv(csv_path)
    start_simulation = date(2025, 1, 1)
    end_simulation = date(2025, 6, 30)
    n_envs = 1
    total_timesteps = 500
    n_steps = 32
    run_simulation(
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        n_steps=n_steps,
        learning_rate=3e-4,
        gamma=0.995,
        n_epochs=10,
        policy="MultiInputPolicy",
        eval_frequency=250,
        eval_seeds=EVAL_SEEDS[:1],
        start_simulation=start_simulation,
        end_simulation=end_simulation,
        num_category1_rooms=50,
        num_category2_rooms=50,
        num_category3_rooms=50,
        seed=SEEDS[0],
        use_subprocesses=False,  # cProfile only sees the main process
    )


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()  # esegui il training sotto profiler

    profiler.disable()

    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    package_path = str(Path(__file__).resolve().parents[1]) + os.sep
    stats.sort_stats("tottime").print_stats(re.escape(package_path), 40)
    print(s.getvalue())

    stats.dump_stats("training_profile.prof")
