from pricing_advertising.policies.policy import Policy


class RandomPolicy(Policy):
    def __init__(self, action_space: int) -> None:
        super().__init__()
        self.action_space = action_space

    def select_action(self) -> int:
        return self.action_space.sample()
