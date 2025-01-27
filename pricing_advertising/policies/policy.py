from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def sample_action(self) -> int:
        pass
