from enum import Enum


class BaseEnum(Enum):
    @classmethod
    def values(cls) -> list:
        """
        Returns a list of values for all members of the enumeration.

        Returns:
            list: A list containing the values of all members of the enumeration.
        """
        return [e.value for e in cls]

    @classmethod
    def names(cls) -> list:
        """
        Returns a list of names of the enumeration members.

        Returns: List of names of the enumeration members.
        """
        return [e.name for e in cls]

    @classmethod
    def has_value(cls, value: int) -> bool:
        """
        Check if the given integer value exists in the enumeration.

        Args:
            value (int): The integer value to check.

        Returns:
            bool: True if the value exists in the enumeration, False otherwise.
        """
        return value in cls.values()

    @classmethod
    def has_name(cls, name: str) -> bool:
        """
        Check if the given name exists in the enumeration.

        Args:
            name (str): The name to check.

        Returns:
            bool: True if the name exists in the enumeration, False otherwise.
        """
        return name in cls.names()

    @classmethod
    def get_name(cls, value: int) -> str:
        """
        Get the name of the enumeration member with the given value.

        Args:
            value (int): The value of the enumeration member.

        Returns:
            str: The name of the enumeration member with the given value.
        """
        for e in cls:
            if e.value == value:
                return e.name
        raise ValueError(f"No member found with value {value}")

    @classmethod
    def get_value(cls, name: str) -> int:
        """
        Get the value of the enumeration member with the given name.

        Args:
            name (str): The name of the enumeration member.

        Returns:
            int: The value of the enumeration member with the given name.
        """
        for e in cls:
            if e.name == name:
                return e.value
        raise ValueError(f"No member found with name {name}")

    # Methods for readability
    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}: {self.value}"
