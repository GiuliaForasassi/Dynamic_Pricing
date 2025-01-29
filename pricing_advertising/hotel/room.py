from datetime import date, timedelta
from enum import Enum


class RoomCategory(Enum):
    CATEGORY1 = 1
    CATEGORY2 = 2
    CATEGORY3 = 3


class Room:
    def __init__(
        self, room_id: int, category: RoomCategory, start_simulation: date, end_simulation: date
    ) -> None:
        self.room_id = room_id  # id of the room
        self.category = category  # category of the room
        self.start_simulation = start_simulation
        self.end_simulation = end_simulation

        # availability of the room in the simulation period, initially all days are available
        self.availability = {
            (start_simulation + timedelta(days=i)): True
            for i in range((end_simulation - start_simulation).days + 1)
        }

    # Check if the room is available for a given date
    def is_available(self, t_reservation: date) -> bool:
        return self.availability[t_reservation]

    # Check if the room is available for more days
    def is_available_days(self, t_reservation: date, days_stay: int) -> bool:
        return all(self.is_available(t_reservation + timedelta(days=i)) for i in range(days_stay))

    # Reserve the room for a given date
    def reserve(self, t_reservation: date) -> bool:
        if self.is_available(t_reservation):
            self.availability[t_reservation] = False
            return True
        else:
            return False

    def reserve_days(self, t_reservation: date, days_stay: int) -> bool:
        if self.is_available_days(t_reservation, days_stay):
            for i in range(days_stay):
                self.reserve(t_reservation + timedelta(days=i))
            return True
        else:
            return False
