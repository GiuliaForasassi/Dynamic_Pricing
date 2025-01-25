from datetime import date
from random import random

from pricing_advertising.hotel.room import Room


class Client:
    def __init__(
        self, t_reservation: date, days_stay: int, num_guests: int
    ) -> None:
        """
        Initialize the client object.

        Args:
            t_reservation (date): Date of reservation.
            days_stay (int): Number of days the client will stay.
            num_guests (int): Number of guests in the reservation.
        """
        self.t_reservation = t_reservation
        self.days_stay = days_stay
        self.num_guests = num_guests

    # proposed_room è la stanza che popongo se quella chiesta dal client non è disponibile
    # TODO: manage proposed_room
    def accept(self, total_price: float, proposed_room: Room) -> bool:
        if total_price < 100:
            return random() < 0.8  # True with 80% probability
        else:
            return random() < 0.2  # True with 20% probability

    def __str__(self) -> str:
        return f"Reservation: {self.t_reservation} | Stay: {self.days_stay} days | Guests: {self.num_guests}"

    def __repr__(self):
        return self.__str__()
