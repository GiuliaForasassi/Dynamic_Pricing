from pricing_advertising.hotel.room import Room


class Hotel:
    def __init__(
        self, num_total_rooms: int, num_category1: int, num_category2: int, num_category3: int
    ) -> None:
        # Initialize the hotel
        self.num_total_rooms = num_total_rooms
        self.num_category1 = num_category1
        self.num_category2 = num_category2
        self.num_category3 = num_category3
        self.rooms = {"category1": [], "category2": [], "category3": []}

    def add_room(self, room: Room) -> None:
        if room.category == 1:
            self.rooms["category1"].append(room.room_id)
        if room.category == 2:
            self.rooms["category2"].append(room.room_id)
        elif room.category == 3:
            self.rooms["category3"].append(room.room_id)


def create_hotel(
    num_total_rooms: int, num_category1: int, num_category2: int, num_category3: int
) -> Hotel:
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
            category = 1
        elif i < num_category1 + num_category2:
            category = 2
        else:
            category = 3
        room = Room(f"room{i}", category)
        hotel.add_room(room)
    return hotel
