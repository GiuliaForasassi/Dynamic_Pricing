from pricing_advertising.hotel.room import Room


class Hotel:
    """
    A class to represent a hotel with dynamic pricing for its rooms.
    """

    def __init__(self, total_rooms: int) -> None:
        """
        Initialize the hotel object.

        Args:
            num_category1 (int): Number of rooms in category 1 (1 guest).
            num_category2 (int): Number of rooms in category 2 (2 guests).
            num_category3 (int): Number of rooms in category 3 (>= 3 guests).
        """
        self.rooms = []
        self.total_rooms = total_rooms

    # Add a room to the hotel
    def add_room(
        self, room_id: int, category: int, available: bool = True
    ) -> None:
        """
        Add a room to the hotel.

        :param : Unique identifier for the room (e.g., number or code).
        :param category: Category of the room (1, 2, or 3).
        :param available: Boolean indicating if the room is available.
        """
        if len(self.rooms) >= self.total_rooms:
            raise Exception(
                "Cannot add more rooms. Hotel is at full capacity."
            )
        room = Room(room_id, category, available)
        self.rooms.append(room)

    def book_room(self, category: int) -> None:
        """
        Book a room in the specified category if available.

        Args:
            category (str): Room category to book.

        Returns:
            str: Confirmation message of the booking.

        Raises:
            ValueError: If the room category is invalid or no rooms are available.
        """

        # Find the first available room in the specified category
        for room in self.rooms:
            if room.get_category() == category and room.get_available():
                room.set_available(False)
                return f"Room {room.get_room_id()} booked successfully."

        raise ValueError(
            f"No available rooms in category {category}. Please try another category."
        )

    def get_all_available_rooms(self) -> list:
        """
        Get a list of all available rooms in the hotel.

        Returns:
            list: List of available rooms.
        """
        return [room for room in self.rooms if room.get_available()]

    def get_available_rooms_per_category(self, category: int) -> list:
        """
        Get a list of available rooms in the specified category.

        Args:
            category (int): Room category to filter by.

        Returns:
            list: List of available rooms in the specified category.
        """
        if category:
            return [
                room
                for room in self.rooms
                if room.get_category() == category and room.get_available()
            ]
        raise ValueError("Please specify a valid room category.")
