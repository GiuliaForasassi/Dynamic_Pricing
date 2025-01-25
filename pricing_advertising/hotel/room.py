class Room:  # noqa: INP001
    """
    A class to represent a room category in the hotel.
    """

    def __init__(self, room_id: str, category: int, available: bool = True):
        """
        Initialize a Room object.

        :param room_id: Unique identifier for the room (e.g., number or code).
        :param category: Category of the room (1, 2, or 3).
        :param available: Boolean indicating if the room is available.
        """
        self.room_id = room_id
        self.category = category
        self.capacity = self.get_capacity(category)
        self.available = available

    @staticmethod
    def get_capacity(category: int) -> int:
        """
        Get the capacity of a room based on its category.

        :param category: Category of the room (1, 2, or 3).
        :return: Capacity of the room.
        """
        if category == 1:
            return 1
        elif category == 2:
            return 2
        elif category == 3:
            return 3
        else:
            raise ValueError("Invalid room category. Must be 1, 2, or 3.")

    def __str__(self) -> str:
        return f"Room {self.room_id} | Category: {self.category} | Capacity: {self.capacity} | Available: {'Yes' if self.available else 'No'}"

    # Getter methods
    def get_room_id(self) -> str:
        return self.room_id

    def get_available(self) -> bool:
        return self.available

    def get_category(self) -> int:
        return self.category

    # Setter methods
    def set_available(self, available: bool) -> None:
        self.available = available

    def set_category(self, category: int) -> None:
        self.category = category
