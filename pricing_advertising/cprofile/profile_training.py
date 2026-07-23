import cProfile
import io
import pstats
from datetime import date
from pathlib import Path

from pricing_advertising.pricing_model.bar_table import set_bar_table_from_csv
from pricing_advertising.training.ppo_training import run_simulation
from pricing_advertising.utils.constants import SEEDS


def main() -> None:
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
        eval_episodes=1,
        start_simulation=start_simulation,
        end_simulation=end_simulation,
        num_category1_rooms=50,
        num_category2_rooms=50,
        num_category3_rooms=50,
        seed=SEEDS[0],
    )


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()  # esegui il training sotto profiler

    profiler.disable()

    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs().sort_stats("tottime").print_stats("pricing_advertising")
    print(s.getvalue())

    stats.dump_stats("training_profile.prof")
