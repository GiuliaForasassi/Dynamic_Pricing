from datetime import date, timedelta


class Room:
    def __init__(
        self, room_id: int, category: int, start_simulation: date, end_simulation: date
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
    def is_available(self, t: date) -> bool:
        return self.availability[t]

    # Reserve the room for a given date
    def reserve(self, t_reservation: date) -> None:
        self.availability[t_reservation] = False

    def reserve_days(self, t_reservation: date, days_stay: int) -> None:
        for i in range(days_stay):
            self.reserve(t_reservation + timedelta(days=i))
