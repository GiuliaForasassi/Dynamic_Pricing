from datetime import date
from random import random

from pricing_advertising.hotel.room import Room


class Client:
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
