from datetime import date

from pricing_advertising.training.ppo_training import run_simulation
from pricing_advertising.utils.constants import SEEDS

if __name__ == "__main__":
    # choose values; must be --> n_steps % n_envs == 0
    final_mean = run_simulation(
        total_cycles=10,
        steps_per_cycle=10000,
        n_envs=2,
        n_steps=256,
        learning_rate=1e-4,
        gamma=0.99,
        n_epochs=1,
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
