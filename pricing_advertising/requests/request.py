from datetime import date


class Request:
    def __init__(self, t_reservation: date, days_stay: int, num_guests: int) -> None:
        self.t_reservation = t_reservation
        self.days_stay = days_stay
        self.num_guests = num_guests

    def __str__(self) -> str:
        return f"Reservation: {self.t_reservation} | Stay: {self.days_stay} days | Guests: {self.num_guests}"

    def __repr__(self):
        return self.__str__()
