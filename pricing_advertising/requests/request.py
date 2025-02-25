from datetime import date
from random import random

from pricing_advertising.enums.enums import RoomCategory


class Request:
    def __init__(self, t_reservation: date, days_stay: int, num_guests: int) -> None:
        self.t_reservation = t_reservation
        self.days_stay = days_stay
        self.num_guests = num_guests

    def accept(self, total_price: float) -> bool:
        """
        Determines whether to accept the proposed room based on the total price.

        Args:
            total_price (float): The total price of the proposed room.
            proposed_room (Room): The proposed room object.

        Returns:
            bool: True if the proposed room is accepted, False otherwise.
        """
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

    def get_t_reservation(self) -> date:
        """
        Retrieve the reservation date.

        Returns:
            date: The reservation date.
        """
        return self.t_reservation

    def get_days_stay(self) -> int:
        """
        Retrieve the number of days of stay.

        Returns:
            int: The number of days of stay.
        """
        return self.days_stay
