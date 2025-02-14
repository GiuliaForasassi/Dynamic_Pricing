from datetime import date, timedelta
from random import random

from pricing_advertising.clients import client
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

    def generate_random_price(self, base_price: float, variability: float = 0.2) -> float:
        min_price = base_price * (1 - variability)
        max_price = base_price * (1 + variability)
        random_price = random.uniform(min_price, max_price)
        return round(random_price, 2)

    def book_room(self, category: RoomCategory, t_reservation: date, days_stay: int) -> Room:
        for room in self.rooms[category]:
            if room.is_available_days(t_reservation, days_stay):
                proposed_price = self.generate_random_price(100)
                if client.accept(proposed_price, room):
                    for i in range(days_stay):
                        room.availability[t_reservation + timedelta(days=i)] = False
                    return room
                else:
                    raise ValueError("Client refused the price")
        # If no room is available in the requested category
        if not self.rooms[category] and category != RoomCategory.CATEGORY3:
            alternative_category = RoomCategory(category.value + 1)
            return self.book_room(alternative_category, t_reservation, days_stay)
        raise ValueError("No room available")
