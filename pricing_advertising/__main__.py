from datetime import date
from pathlib import Path

from pricing_advertising.pricing_model.bar_table import set_bar_table_from_csv
from pricing_advertising.training.ppo_training import run_simulation
from pricing_advertising.utils.constants import SEEDS

if __name__ == "__main__":
    # Load bar table
    pkg_dir = Path(__file__).resolve().parent
    csv_path = pkg_dir / "data" / "bar_table.csv"
    set_bar_table_from_csv(csv_path)

    # choose values; must be --> n_steps % n_envs == 0
    final_mean = run_simulation(
        total_cycles=10,
        # number of timesteps of training to do in the learn cycle
        steps_per_cycle=10240,
        n_envs=2,
        n_steps=256,
        learning_rate=1e-4,
        gamma=0.99,
        n_epochs=5,
        policy="MultiInputPolicy",
        eval_episodes=5,
        start_simulation=date(2025, 1, 1),
        end_simulation=date(2025, 12, 1),
        num_category1_rooms=50,
        num_category2_rooms=50,
        num_category3_rooms=50,
        seed=SEEDS[0],
    )
    print(f"Final eval mean reward: {final_mean:.3f}")
