from enum import Enum


class BaseEnum(Enum):
    @classmethod
    def values(cls) -> set:
        """
        Returns a set of values for all members of the enumeration.

        Returns:
            set: A set containing the values of all members of the enumeration.
        """
        return {e.value for e in cls}

    @classmethod
    def names(cls) -> set:
        """
        Returns a set of names of the enumeration members.

        Returns: set of names of the enumeration members.
        """
        return {e.name for e in cls}

    # Methods for readability
    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}: {self.value}"
