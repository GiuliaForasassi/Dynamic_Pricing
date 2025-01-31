from datetime import date

from pricing_advertising.hotel.room import Room, RoomCategory


class Hotel:
    def __init__(
        self,
        num_total_rooms: int,
        num_category1_rooms: int,
        num_category2_rooms: int,
        num_category3_rooms: int,
    ) -> None:
        # Initialize the hotel
        self.num_total_rooms = num_total_rooms
        self.num_category1_rooms = num_category1_rooms
        self.num_category2_rooms = num_category2_rooms
        self.num_category3_rooms = num_category3_rooms
        self.rooms: dict[RoomCategory, list[Room]] = {
            RoomCategory.CATEGORY1: [],
            RoomCategory.CATEGORY2: [],
            RoomCategory.CATEGORY3: [],
        }

    def add_room(self, room: Room) -> None:
        self.rooms[room.category].append(room)

    @staticmethod
    def create_hotel(
        num_total_rooms: int, num_category1: int, num_category2: int, num_category3: int
    ) -> "Hotel":
        # Validate input parameters
        if num_total_rooms != num_category1 + num_category2 + num_category3:
            raise ValueError(
                "The total number of rooms must be equal to the sum of the rooms in each category"
            )
        # Create the hotel
        hotel = Hotel(num_total_rooms, num_category1, num_category2, num_category3)
        # Populate the hotel with rooms
        for i in range(num_total_rooms):
            if i < num_category1:
                category = RoomCategory.CATEGORY1
            elif i < num_category1 + num_category2:
                category = RoomCategory.CATEGORY2
            else:
                category = RoomCategory.CATEGORY3
            room = Room(f"room{i}", category)
            hotel.add_room(room)
        return hotel

    # TODO: add price management
    # TODO: add another room proposal if the requested room is not available
    def book_room(self, category: RoomCategory, t_reservation: date, days_stay: int) -> Room | None:
        # If a room of the given category is available, reserve it
        for room in self.rooms[category]:
            if room.reserve_days(t_reservation, days_stay):
                return room

        # If no room of the given category is available, propose an alternative room
        alternatives = {
            RoomCategory.CATEGORY1: [RoomCategory.CATEGORY2, RoomCategory.CATEGORY3],
            RoomCategory.CATEGORY2: [RoomCategory.CATEGORY3],
            RoomCategory.CATEGORY3: [],
        }

        return None
