"""Run a configured PPO experiment."""

import argparse
import os
from pathlib import Path

from pydantic import ValidationError

from pricing_advertising.config import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quick.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a configured PPO experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Experiment JSON file")
    parser.add_argument(
        "--check-config", action="store_true", help="Validate without running training"
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (OSError, ValueError, ValidationError) as error:
        parser.error(f"Invalid configuration {args.config}: {error}")

    print(f"Configuration: {args.config.resolve()}")
    for name, value in config.summary().items():
        print(f"{name} = {value}")
    for warning in config.warnings():
        print(f"Warning: {warning}")
    if args.check_config:
        return

    from pricing_advertising.pricing_model.bar_table import set_bar_table_from_csv
    from pricing_advertising.training.multi_seed_training import run_multi_seed_simulation

    tracking_uri = os.environ.setdefault("MLFLOW_TRACKING_URI", config.run.tracking_uri)
    set_bar_table_from_csv(config.run.bar_table_path)
    resolved_config = config.model_dump(mode="json")
    resolved_config["run"]["tracking_uri"] = tracking_uri
    final_mean_reward = run_multi_seed_simulation(
        **config.runner_kwargs(), resolved_config=resolved_config
    )
    print(
        f"Final eval mean reward (mean over {len(config.run.seeds)} seeds): {final_mean_reward:.3f}"
    )


if __name__ == "__main__":
    main()
