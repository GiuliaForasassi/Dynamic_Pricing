from datetime import date

from loguru import logger

from pricing_advertising.enums.enums import RoomCategory


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

    def get_category(self) -> RoomCategory:
        """
        Determines the room category based on the number of guests in the request.

        Args:
            request (Request): The request object containing the number of guests.

        Returns:
            RoomCategory: The category of the room based on the number of guests.

        Raises:
            ValueError: If the number of guests is less than 1 or greater than 3.
        """
        num_guests = self.num_guests
        if num_guests < 1:
            raise ValueError("The number of guests must be greater than 0.")
        if num_guests == 1:
            return RoomCategory.CATEGORY1
        elif num_guests == 2:
            return RoomCategory.CATEGORY2
        elif num_guests == 3:
            return RoomCategory.CATEGORY3
        else:
            raise ValueError("The number of guests {num_guests} must be less than or equal to 3.")


if __name__ == "__main__":
    request = Request(date.today(), 3, 2)
    logger.info(request)
    logger.info(request.get_total_reservation_days())
    logger.info(request.get_category())
