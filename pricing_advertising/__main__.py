import math
import os
from datetime import date
from pathlib import Path

from pricing_advertising.pricing_model.bar_table import set_bar_table_from_csv
from pricing_advertising.training.always_increment_policy import (
    run_always_increment_policy_simulation,
)
from pricing_advertising.training.constant_bar_policy import run_constant_bar_policy_simulation
from pricing_advertising.training.multi_seed_training import run_multi_seed_simulation
from pricing_advertising.training.random_policy import run_random_policy_simulation
from pricing_advertising.utils.constants import (
    EVAL_SEEDS,
    SEEDS,
    TOTAL_TIMESTEPS_1K,
    TOTAL_TIMESTEPS_1M,
    TOTAL_TIMESTEPS_30K,
)

if __name__ == "__main__":
    # Same backend as `make mlflow-ui`; the ./mlruns file store is no longer accepted by MLflow.
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    # ----- Load bar table from CSV ---------
    csv_path = Path("data") / "bar_table.csv"
    set_bar_table_from_csv(csv_path)

    # ------- Simulation parameters ---------
    start_simulation = date(2025, 1, 1)  # Start date of the simulation
    end_simulation = date(2025, 12, 31)  # End date of the simulation
    n_envs = 4  # Number of parallel environments
    total_timesteps = TOTAL_TIMESTEPS_1M  # Total number of timesteps for training
    n_steps = 730  # Number of steps per environment per rollout
    learning_rate = 3e-4  # Learning rate for the optimizer
    gamma = 0.999  # Discount factor for future rewards
    n_epochs = 10  # Number of epochs per update
    batch_size = 365  # Minibatch size; preferably a divisor of n_steps * n_envs
    ent_coef = 0.0  # Entropy coefficient for the loss calculation
    gae_lambda = 0.98  # Bias-variance trade-off for advantage estimation
    clip_range = 0.2  # PPO policy clipping range
    net_arch = dict(pi=[64, 64], vf=[64, 64])  # Hidden layers for actor and critic
    eval_frequency = 10000
    eval_seeds = list(EVAL_SEEDS)
    baseline_eval_episodes = 1
    num_category1_rooms = 100  # Number of rooms in category 1
    num_category2_rooms = 50  # Number of rooms in category 2
    num_category3_rooms = 30  # Number of rooms in category 3
    avg_num_requests = 60  # Average number of requests per day
    allow_category_alternative = True  # Both False disables alternatives (baseline)
    allow_period_alternative = True

    # ----- Derived simulation parameters ---------
    n_eval_envs = len(eval_seeds)
    n_eval_episodes = n_eval_envs
    episode_len = (end_simulation - start_simulation).days + 1
    rollout_size = n_steps * n_envs  # Total number of steps per rollout across all environments
    n_rollouts = math.ceil(total_timesteps / rollout_size)
    effective_timesteps = n_rollouts * rollout_size
    expected_logged_episodes = effective_timesteps / (episode_len * n_envs)

    # ----- Print derived simulation parameters ---------
    print("n_eval_envs =", n_eval_envs)
    print("n_eval_episodes =", n_eval_episodes)
    print("episode_len =", episode_len)
    print("rollout_size =", rollout_size)
    print("n_rollouts =", n_rollouts)
    print("effective_timesteps =", effective_timesteps)
    print("expected_logged_episodes =", expected_logged_episodes)

    # # ----- Run PPO policy simulation: one child run per seed in SEEDS, averaged in the parent ---------
    final_mean_reward = run_multi_seed_simulation(
        run_name="PPO_1M_with_alternatives",
        seeds=SEEDS,
        eval_seeds=eval_seeds,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        n_steps=n_steps,
        learning_rate=learning_rate,
        gamma=gamma,
        n_epochs=n_epochs,
        batch_size=batch_size,
        ent_coef=ent_coef,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        net_arch=net_arch,
        policy="MultiInputPolicy",
        eval_frequency=eval_frequency,
        start_simulation=start_simulation,
        end_simulation=end_simulation,
        num_category1_rooms=num_category1_rooms,
        num_category2_rooms=num_category2_rooms,
        num_category3_rooms=num_category3_rooms,
        price_plot_request_dates=[
            (2025, 2, 28),
            (2025, 5, 15),
            (2025, 8, 1),
            (2025, 11, 12),
        ],
        avg_num_requests=avg_num_requests,
        allow_category_alternative=allow_category_alternative,
        allow_period_alternative=allow_period_alternative,
    )
    print(f"Final eval mean reward (mean over {len(SEEDS)} seeds): {final_mean_reward:.3f}")

    # ----- Run random policy simulation ---------
    # final_mean_reward = run_random_policy_simulation(
    #     run_name="Random_baseline_1M_with_alternatives",
    #     seed=SEEDS[0],
    #     eval_seed=SEEDS[0] + 10_000,
    #     start_simulation=start_simulation,
    #     end_simulation=end_simulation,
    #     num_category1_rooms=num_category1_rooms,
    #     num_category2_rooms=num_category2_rooms,
    #     num_category3_rooms=num_category3_rooms,
    #     avg_num_requests=avg_num_requests,
    #     eval_episodes=baseline_eval_episodes,
    #     allow_category_alternative=allow_category_alternative,
    #     allow_period_alternative=allow_period_alternative,
    #     price_plot_request_dates=[
    #         (2025, 2, 28),
    #         (2025, 5, 15),
    #         (2025, 8, 1),
    #         (2025, 11, 12),
    #     ],
    # )
    # print(f"Random eval mean reward: {final_mean_reward:.3f}")

    # # ----- Run constant BAR policy simulation (never increments) ---------
    # final_mean_reward = run_constant_bar_policy_simulation(
    #     run_name="Constant_bar_baseline_1M_with_alternatives",
    #     seed=SEEDS[0],
    #     eval_seed=SEEDS[0] + 10_000,
    #     start_simulation=start_simulation,
    #     end_simulation=end_simulation,
    #     num_category1_rooms=num_category1_rooms,
    #     num_category2_rooms=num_category2_rooms,
    #     num_category3_rooms=num_category3_rooms,
    #     avg_num_requests=avg_num_requests,
    #     eval_episodes=baseline_eval_episodes,
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
    #     run_name="Always_increment_baseline_1M_with_alternatives",
    #     seed=SEEDS[0],
    #     eval_seed=SEEDS[0] + 10_000,
    #     start_simulation=start_simulation,
    #     end_simulation=end_simulation,
    #     num_category1_rooms=num_category1_rooms,
    #     num_category2_rooms=num_category2_rooms,
    #     num_category3_rooms=num_category3_rooms,
    #     avg_num_requests=avg_num_requests,
    #     eval_episodes=baseline_eval_episodes,
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
