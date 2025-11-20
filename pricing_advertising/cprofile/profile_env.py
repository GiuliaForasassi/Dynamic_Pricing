import cProfile
import io
import pstats
from datetime import date
from pathlib import Path

from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.hotel.hotel import Hotel
from pricing_advertising.pricing_model.bar_table import set_bar_table_from_csv
from pricing_advertising.utils.constants import SEEDS


def make_env() -> PricingEnv:
    """
    Crea un env come fai nel training.
    Cerca di usare parametri "reali" ma magari un po' più piccoli.
    """

    # Esempio di costruzione Hotel (adatta ai tuoi costruttori reali)
    hotel = Hotel(
        num_category1_rooms=10,
        num_category2_rooms=10,
        num_category3_rooms=10,
    )

    # Load bar table
    csv_path = Path("data") / "bar_table.csv"
    set_bar_table_from_csv(csv_path)

    env = PricingEnv(
        start_simulation=date(2025, 1, 1),
        end_simulation=date(2025, 1, 31),
        hotel=hotel,
        avg_num_requests=None,
        request_generator=None,
        acceptance_model=None,
        seed=SEEDS[0],
    )
    return env


def run_env_loop(num_episodes: int = 5, max_steps_per_ep: int = 365) -> None:
    env = make_env()

    for _ in range(num_episodes):
        obs, info = env.reset(seed=SEEDS[0])
        done = False
        step_count = 0

        while not done and step_count < max_steps_per_ep:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

    env.close()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    run_env_loop(num_episodes=5, max_steps_per_ep=365)

    profiler.disable()

    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs().sort_stats("tottime").print_stats(40)
    print("\n=== Top 40 funzioni per tottime (env only) ===")
    print(s.getvalue())

    stats.dump_stats("env_profile.prof")
