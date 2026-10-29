"""Episode returns of the training-free baselines on the evaluation calendar."""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import gymnasium as gym

from pricing_advertising.environment.env import PricingEnv
from pricing_advertising.training.always_increment_policy import AlwaysIncrementPolicy
from pricing_advertising.training.constant_bar_policy import ConstantBarPolicy
from pricing_advertising.training.random_policy import RandomPolicy

type _BarUpdateMode = Literal["never", "always"] | None

# name -> (policy factory, bar_update_mode)
BASELINES: dict[str, tuple[Callable[[gym.spaces.Space, int], Any], _BarUpdateMode]] = {
    "constant_bar": (lambda space, _seed: ConstantBarPolicy(space), "never"),
    "always_increment": (lambda space, _seed: AlwaysIncrementPolicy(space), "always"),
    "random": (lambda space, seed: RandomPolicy(space, seed), None),
}


def evaluate_baseline_returns(
    env_kwargs: dict[str, Any], *, eval_seeds: Sequence[int], seed: int
) -> dict[str, list[float]]:
    """Raw episode revenue of each baseline: one episode per evaluation seed, as in the eval envs."""
    returns: dict[str, list[float]] = {}
    for name, (make_policy, bar_update_mode) in BASELINES.items():
        returns[name] = []
        policy = None
        for eval_seed in eval_seeds:
            env = PricingEnv(**env_kwargs, seed=eval_seed, bar_update_mode=bar_update_mode)
            try:
                if policy is None:
                    policy = make_policy(env.action_space, seed)
                observation, _ = env.reset(seed=eval_seed)
                total = 0.0
                done = False
                while not done:
                    action, _ = policy.predict(observation)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                    done = terminated or truncated
                returns[name].append(total)
            finally:
                env.close()
    return returns
