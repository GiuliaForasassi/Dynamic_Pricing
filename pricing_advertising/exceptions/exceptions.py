from datetime import date

from pricing_advertising.exceptions.base_exceptions import (
    BaseBookingError,
    BaseEnvError,
    BaseHotelError,
    BaseImplementationError,
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
        message = f"The provided period ({start_period}, {end_period}) is not valid. Period must be included within {start_simulation} and {end_simulation} days."
        super().__init__(message=message)


class ClientRefusedError(BaseBookingError):
    def __init__(self) -> None:
        message = "The client refused the offer."
        super().__init__(message=message)


# endregion


class CustomNotImplementedError(BaseImplementationError):
    def __init__(self) -> None:
        message = "This method is not implemented yet."
        super().__init__(message=message)


class RequestGeneratorError(BaseRequestError):
    def __init__(self, remaining_days: int) -> None:
        message = f"The {remaining_days} exceeds the total number of days."
        super().__init__(message=message)


class RoomUnavailableError(BaseHotelError):
    """Exception raised when no room is available for requested category"""


class NoneAlternativeError(BaseHotelError):
    """Exception raised when there is no alternative room available"""


class EnvironmentInitializationError(BaseEnvError):
    """Exception raised when there is a problem during Environment
    service initialization"""
