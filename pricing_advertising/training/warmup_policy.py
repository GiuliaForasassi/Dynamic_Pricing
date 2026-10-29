"""Frozen copy of the policy being trained, used by PricingEnv to simulate its warmup days."""

from typing import Any

import cloudpickle
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv


class FrozenPolicy:
    """Snapshot of a policy: later training of the original does not change it.

    Observations must be the raw ones (no VecNormalize observation scaling). Pickle-safe, so
    it can be sent to SubprocVecEnv workers (SB3 policies hold closures that plain pickle rejects).
    """

    def __init__(self, policy: BasePolicy, *, deterministic: bool) -> None:
        self._payload = cloudpickle.dumps(policy)
        self._deterministic = deterministic
        self._policy: BasePolicy | None = None

    def __call__(self, observation: dict[str, Any]) -> np.ndarray:
        if self._policy is None:
            self._policy = cloudpickle.loads(self._payload).to("cpu")  # noqa: S301
            self._policy.set_training_mode(False)
        action, _ = self._policy.predict(observation, deterministic=self._deterministic)
        return action

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_policy": None}


def sync_warmup_policy(vec_env: VecEnv, policy: BasePolicy, *, deterministic: bool) -> None:
    """Make the warmup of every environment of `vec_env` use a frozen copy of `policy`."""
    vec_env.env_method("set_warmup_policy", FrozenPolicy(policy, deterministic=deterministic))


class WarmupPolicyCallback(BaseCallback):
    """Refreshes the training envs' warmup policy at the start of every rollout."""

    def _on_rollout_start(self) -> None:
        sync_warmup_policy(self.training_env, self.model.policy, deterministic=False)

    def _on_step(self) -> bool:
        return True
