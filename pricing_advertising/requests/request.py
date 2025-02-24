from datetime import date
from random import random

from pricing_advertising.hotel.room import Room, RoomCategory


class Request:
    def __init__(self, t_reservation: date, days_stay: int, num_guests: int) -> None:
        self.t_reservation = t_reservation
        self.days_stay = days_stay
        self.num_guests = num_guests

    def accept(self, total_price: float, proposed_room: Room) -> bool:
        if total_price < 100:
            return random() < 0.8  # True with 80% probability
        else:
            return random() < 0.2  # True with 20% probability

    def __str__(self) -> str:
        return f"Reservation: {self.t_reservation} | Stay: {self.days_stay} days | Guests: {self.num_guests}"

    def __repr__(self):
        return self.__str__()

    def get_category(self) -> RoomCategory:
        """
        Determines the room category based on the number of guests.

        Returns:
            RoomCategory: The category of the room based on the number of guests.
                - CATEGORY1: If there is 1 guest.
                - CATEGORY2: If there are 2 guests.
                - CATEGORY3: If there 3 guests.
        """
        if self.num_guests == 1:
            return RoomCategory.CATEGORY1
        elif self.num_guests == 2:
            return RoomCategory.CATEGORY2
        else:
            return RoomCategory.CATEGORY3
