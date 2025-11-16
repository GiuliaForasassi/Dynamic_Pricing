from datetime import date
from pathlib import Path

from pricing_advertising.pricing_model.bar_table import set_bar_table_from_csv
from pricing_advertising.training.ppo_training import run_simulation
from pricing_advertising.utils.constants import SEEDS

if __name__ == "__main__":
    # Load bar table
    csv_path = Path("data") / "bar_table.csv"
    set_bar_table_from_csv(csv_path)

    start_simulation = date(2025, 2, 1)
    end_simulation = date(2025, 2, 8)
    simulation_days = (end_simulation - start_simulation).days
    simulations_per_cycle_per_subprocess = 3

    # choose values; must be --> n_steps % n_envs == 0
    final_mean = run_simulation(
        total_cycles=1,
        # number of timesteps of training to do in the learn cycle
        steps_per_cycle=simulation_days * simulations_per_cycle_per_subprocess,
        n_envs=4,
        n_steps=4,
        learning_rate=1e-4,
        gamma=0.99,
        n_epochs=3,
        policy="MultiInputPolicy",
        eval_episodes=1,
        start_simulation=start_simulation,
        end_simulation=end_simulation,
        num_category1_rooms=5,
        num_category2_rooms=5,
        num_category3_rooms=5,
        seed=SEEDS[0],
    )
    print(f"Final eval mean reward: {final_mean:.3f}")
