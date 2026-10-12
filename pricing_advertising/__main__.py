import math
from datetime import date
from pathlib import Path

from pricing_advertising.pricing_model.bar_table import set_bar_table_from_csv
from pricing_advertising.training.always_increment_policy import (
    run_always_increment_policy_simulation,
)
from pricing_advertising.training.constant_bar_policy import run_constant_bar_policy_simulation
from pricing_advertising.training.ppo_training import run_simulation
from pricing_advertising.training.random_policy import run_random_policy_simulation
from pricing_advertising.utils.constants import SEEDS, TOTAL_TIMESTEPS_1M

if __name__ == "__main__":
    # ----- Load bar table from CSV ---------
    csv_path = Path("data") / "bar_table.csv"
    set_bar_table_from_csv(csv_path)

    # ------- Simulation parameters ---------
    start_simulation = date(2025, 1, 1)  # Start date of the simulation
    end_simulation = date(2025, 12, 31)  # End date of the simulation
    n_envs = 4  # Number of parallel environments
    total_timesteps = TOTAL_TIMESTEPS_1M  # Total number of timesteps for training
    n_steps = 500  # Number of steps per environment per rollout
    learning_rate = 3e-4  # Learning rate for the optimizer
    gamma = 0.995  # Discount factor for future rewards
    n_epochs = 10  # Number of epochs per update
    batch_size = 250  # Minibatch size; preferably a divisor of n_steps * n_envs
    ent_coef = 0.001  # Entropy coefficient encouraging exploration
    gae_lambda = 0.95  # Bias-variance trade-off for advantage estimation
    clip_range = 0.2  # PPO policy clipping range
    net_arch = dict(pi=[64, 64], vf=[64, 64])  # Hidden layers for actor and critic
    eval_frequency = 10000  # Frequency of evaluation during training
    eval_episodes = 1  # Number of episodes for each evaluation
    num_category1_rooms = 100  # Number of rooms in category 1
    num_category2_rooms = 50  # Number of rooms in category 2
    num_category3_rooms = 30  # Number of rooms in category 3
    avg_num_requests = 60  # Average number of requests per day
    allow_category_alternative = False  # Both False disables alternatives (baseline)
    allow_period_alternative = False

    # ----- Derived simulation parameters ---------
    episode_len = (end_simulation - start_simulation).days + 1
    rollout_size = n_steps * n_envs  # Total number of steps per rollout across all environments
    n_rollouts = math.ceil(total_timesteps / rollout_size)
    effective_timesteps = n_rollouts * rollout_size
    expected_logged_episodes = effective_timesteps / (episode_len * n_envs)

    # ----- Print derived simulation parameters ---------
    print("episode_len =", episode_len)
    print("rollout_size =", rollout_size)
    print("n_rollouts =", n_rollouts)
    print("effective_timesteps =", effective_timesteps)
    print("expected_logged_episodes =", expected_logged_episodes)

    # # ----- Run PPO policy simulation ---------
    # final_mean_reward = run_simulation(
    #     run_name="PPO_1M_no_alternatives",
    #     total_timesteps=total_timesteps,
    #     n_envs=n_envs,
    #     n_steps=n_steps,
    #     learning_rate=learning_rate,
    #     gamma=gamma,
    #     n_epochs=n_epochs,
    #     batch_size=batch_size,
    #     ent_coef=ent_coef,
    #     gae_lambda=gae_lambda,
    #     clip_range=clip_range,
    #     net_arch=net_arch,
    #     policy="MultiInputPolicy",
    #     eval_episodes=eval_episodes,
    #     eval_frequency=eval_frequency,
    #     start_simulation=start_simulation,
    #     end_simulation=end_simulation,
    #     num_category1_rooms=num_category1_rooms,
    #     num_category2_rooms=num_category2_rooms,
    #     num_category3_rooms=num_category3_rooms,
    #     price_plot_request_dates=[
    #         (2025, 2, 28),
    #         (2025, 5, 15),
    #         (2025, 8, 1),
    #         (2025, 11, 12),
    #     ],
    #     seed=SEEDS[0],
    #     avg_num_requests=avg_num_requests,
    #     allow_category_alternative=allow_category_alternative,
    #     allow_period_alternative=allow_period_alternative,
    #     eval_seed=SEEDS[0] + 10_000,
    # )
    # print(f"Final eval mean reward: {final_mean_reward:.3f}")

    # ----- Run random policy simulation ---------
    final_mean_reward = run_random_policy_simulation(
        run_name="Random_baseline_1M",
        seed=SEEDS[0],
        eval_seed=SEEDS[0] + 10_000,
        start_simulation=start_simulation,
        end_simulation=end_simulation,
        num_category1_rooms=num_category1_rooms,
        num_category2_rooms=num_category2_rooms,
        num_category3_rooms=num_category3_rooms,
        avg_num_requests=avg_num_requests,
        eval_episodes=eval_episodes,
        allow_category_alternative=allow_category_alternative,
        allow_period_alternative=allow_period_alternative,
        price_plot_request_dates=[
            (2025, 2, 28),
            (2025, 5, 15),
            (2025, 8, 1),
            (2025, 11, 12),
        ],
    )
    print(f"Random eval mean reward: {final_mean_reward:.3f}")

    # # ----- Run constant BAR policy simulation (never increments) ---------
    # final_mean_reward = run_constant_bar_policy_simulation(
    #     run_name="Constant_bar_baseline_1M",
    #     seed=SEEDS[0],
    #     eval_seed=SEEDS[0] + 10_000,
    #     start_simulation=start_simulation,
    #     end_simulation=end_simulation,
    #     num_category1_rooms=num_category1_rooms,
    #     num_category2_rooms=num_category2_rooms,
    #     num_category3_rooms=num_category3_rooms,
    #     avg_num_requests=avg_num_requests,
    #     eval_episodes=eval_episodes,
    #     allow_category_alternative=allow_category_alternative,
    #     allow_period_alternative=allow_period_alternative,
    #     price_plot_request_dates=[
    #         (2025, 2, 28),
    #         (2025, 5, 15),
    #         (2025, 8, 1),
    #         (2025, 11, 12),
    #     ],
    # )
    # print(f"Constant BAR eval mean reward: {final_mean_reward:.3f}")

    # ----- Run always-increment policy simulation (+1 BAR level every step) ---------
    # final_mean_reward = run_always_increment_policy_simulation(
    #     run_name="Always_increment_baseline_1M",
    #     seed=SEEDS[0],
    #     eval_seed=SEEDS[0] + 10_000,
    #     start_simulation=start_simulation,
    #     end_simulation=end_simulation,
    #     num_category1_rooms=num_category1_rooms,
    #     num_category2_rooms=num_category2_rooms,
    #     num_category3_rooms=num_category3_rooms,
    #     avg_num_requests=avg_num_requests,
    #     eval_episodes=eval_episodes,
    #     allow_category_alternative=allow_category_alternative,
    #     allow_period_alternative=allow_period_alternative,
    #     price_plot_request_dates=[
    #         (2025, 2, 28),
    #         (2025, 5, 15),
    #         (2025, 8, 1),
    #         (2025, 11, 12),
    #     ],
    # )
    # print(f"Always increment eval mean reward: {final_mean_reward:.3f}")
