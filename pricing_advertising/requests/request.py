from datetime import date

from loguru import logger

from pricing_advertising.enums.enums import RoomCategory


class Request:
    def __init__(
        self, t_request: date, t_reservation: date, days_stay: int, num_guests: int
    ) -> None:
        if t_request > t_reservation:
            raise ValueError(
                f"The request date {t_request} must be before or equal the reservation date {t_reservation}."
            )
        self.t_request = t_request
        self.t_reservation = t_reservation
        self.days_stay = days_stay
        self.num_guests = num_guests
        self.proposed_t_reservation: date | None = None
        self.proposed_days_stay: int | None = None

    def __str__(self) -> str:
        return f"Request: {self.t_request} | Reservation: {self.t_reservation} | Stay: {self.days_stay} days | Guests: {self.num_guests}"

    @property
    def total_reservation_days(self) -> int:
        """
        Get the total number of reservation days for a given request.

        Args:
            request (Request): The request object containing the reservation details.

        Returns:
            int: The total number of days for the request.
        """
        return self.days_stay + 1  # +1 to include the t_res day

    @property
    def category(self) -> RoomCategory:
        """
        Determines the room category based on the number of guests.

        Returns:
            RoomCategory: The category of the room corresponding to the number of guests.

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
            raise ValueError(f"The number of guests {num_guests} must be less than or equal to 3.")

    @category.setter
    def category(self, value: RoomCategory) -> None:
        """
        Set the room category based on the number of guests.

        Args:
            value (RoomCategory): The room category to set.
        """
        if not isinstance(value, RoomCategory):
            raise ValueError("Invalid room category.")
        # Reverse the getter: setting the category updates num_guests
        if value == RoomCategory.CATEGORY1:
            self.num_guests = 1
        elif value == RoomCategory.CATEGORY2:
            self.num_guests = 2
        elif value == RoomCategory.CATEGORY3:
            self.num_guests = 3
        else:
            raise ValueError(f"Unsupported RoomCategory: {value}.")

    @property
    def advance(self) -> int:
        """
        Get the number of days in advance the reservation was made.

        Args:
            request (Request): The request object containing the reservation details.

        Returns:
            int: The number of days in advance the reservation was made.
        """
        return (self.t_reservation - self.t_request).days


if __name__ == "__main__":
    today = date.today()
    request = Request(
        t_request=date(today.year, today.month, today.day - 1),
        t_reservation=today,
        days_stay=3,
        num_guests=2,
    )
    logger.info(request)
    logger.info(request.total_reservation_days)
    logger.info(request.category)

    assert request.total_reservation_days == 4
    assert request.category == RoomCategory.CATEGORY2
