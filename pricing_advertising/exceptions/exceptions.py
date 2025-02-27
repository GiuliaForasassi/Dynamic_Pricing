from datetime import date

from pricing_advertising.exceptions.base_exceptions import (
    BookingError,
    EnvError,
    HotelError,
    ImplementationError,
    RequestError,
)


class NotValidPeriodError(BookingError):
    def __init__(
        self,
        start_period: date,
        end_period: date,
        start_simulation: date,
        end_simulation: date,
        **kwargs,
    ) -> None:
        message = f"The provided period ({start_period}, {end_period}) is not valid. Period must be included within {start_simulation} and {end_simulation} days."
        super().__init__(message=message, **kwargs)


class ClientRefusedError(BookingError):
    def __init__(self) -> None:
        message = "The client refused the offer."
        super().__init__(message=message)


class CustomNotImplementedError(ImplementationError):
    def __init__(self) -> None:
        message = "This method is not implemented yet."
        super().__init__(message=message)


class RequestGeneratorError(RequestError):
    def __init__(self, remaining_days: int) -> None:
        message = f"The {remaining_days} exceeds the total number of days."
        super().__init__(message=message)


class RoomUnavailableError(HotelError):
    """Exception raised when no room is available for requested category"""


class NoneAlternativeError(HotelError):
    """Exception raised when there is no alternative room available"""


class EnvironmentInitializationError(EnvError):
    """Exception raised when there is a problem during Environment
    service initialization"""
