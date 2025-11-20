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
    run_simulation(
        # ======= TRAINING SCHEDULE =======
        total_cycles=2,  # es. piccolino per profiling
        steps_per_cycle=30,  # o meno se vuoi
        # ======= PPO/ENV =======
        n_envs=2,
        n_steps=4,
        learning_rate=1e-4,
        gamma=0.99,
        n_epochs=10,
        policy="MultiInputPolicy",
        # ======= EVAL =======
        eval_episodes=3,
        # ======= SIMULAZIONE HOTEL =======
        start_simulation=date(2025, 1, 1),
        end_simulation=date(2025, 1, 31),
        num_category1_rooms=10,
        num_category2_rooms=10,
        num_category3_rooms=10,
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
