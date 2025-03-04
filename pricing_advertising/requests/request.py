from datetime import date


class Request:
    def __init__(self, t_reservation: date, days_stay: int, num_guests: int) -> None:
        self.t_reservation = t_reservation
        self.days_stay = days_stay
        self.num_guests = num_guests

    def __str__(self) -> str:
        return f"Reservation: {self.t_reservation} | Stay: {self.days_stay} days | Guests: {self.num_guests}"

    def get_total_reservation_days(self) -> int:
        """
        Get the total number of reservation days for a given request.

        Args:
            request (Request): The request object containing the reservation details.

        Returns:
            int: The total number of days for the request.
        """
        return self.days_stay + 1  # +1 to include the t_res day
