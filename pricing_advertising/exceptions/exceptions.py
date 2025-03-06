from datetime import date

from pricing_advertising.exceptions.base_exceptions import (
    BaseBookingError,
    BaseEnvError,
    BaseHotelError,
    BaseRequestError,
)


# region Booking Errors
class NotValidPeriodError(BaseBookingError):
    def __init__(
        self,
        start_period: date,
        end_period: date,
        start_simulation: date,
        end_simulation: date,
    ) -> None:
        message = f"The provided period ({start_period}, {end_period}) is not valid. Period must be included within ({start_simulation}, {end_simulation})."
        super().__init__(message)


# endregion


class InvalidReservationWindowError(BaseRequestError):
    def __init__(self, remaining_days: int) -> None:
        message = f"The {remaining_days} exceeds the total number of days."
        super().__init__(message)


class RoomUnavailableError(BaseHotelError):
    def __init__(self, category: int) -> None:
        message = f"Exceeded the number of available rooms for category {category}."
        super().__init__(message)


class NoneAlternativeError(BaseHotelError):
    """Exception raised when there is no alternative room available"""


class EnvironmentInitializationError(BaseEnvError):
    """Exception raised when there is a problem during Environment
    service initialization"""
