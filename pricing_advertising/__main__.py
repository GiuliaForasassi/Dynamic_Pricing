import cProfile
import math
from datetime import date
from pathlib import Path

from pricing_advertising.pricing_model.bar_table import set_bar_table_from_csv
from pricing_advertising.training.ppo_training import run_simulation
from pricing_advertising.utils.constants import SEEDS

if __name__ == "__main__":
    # Load bar table
    csv_path = Path("data") / "bar_table.csv"
    set_bar_table_from_csv(csv_path)

    start_simulation = date(2025, 1, 1)
    end_simulation = date(2025, 12, 31)
    n_envs = 4
    total_timesteps = 100000
    n_steps = 2048

    episode_len = (end_simulation - start_simulation).days
    rollout_size = n_steps * n_envs
    n_rollouts = math.ceil(total_timesteps / rollout_size)
    effective_timesteps = n_rollouts * rollout_size
    expected_logged_episodes = effective_timesteps / (episode_len * n_envs)

    print("episode_len =", episode_len)
    print("rollout_size =", rollout_size)
    print("n_rollouts =", n_rollouts)
    print("effective_timesteps =", effective_timesteps)
    print("expected_logged_episodes =", expected_logged_episodes)

    # choose values; must be --> n_steps % n_envs == 0
    final_mean_reward = run_simulation(
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
    print(f"Final eval mean reward: {final_mean_reward:.3f}")
