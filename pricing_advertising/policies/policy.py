from abc import ABC, abstractmethod


class Policy(ABC):
    # def __init__(self) -> None:
    # self.num_actions = ...
    # self.action_space = ....

    @abstractmethod
    def select_action(self) -> int:
        pass
