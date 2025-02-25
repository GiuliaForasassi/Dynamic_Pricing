class ClientRefusedError(Exception):
    """Exception raised when the client refuses the offer"""


class RequestGeneratorError(Exception):
    """Exception raised when there is a problem with the generation of a request"""


class RoomUnavailableError(Exception):
    """Exception raised when no room is available for requested category"""


class NotValidPeriodError(Exception):
    """Exception raised when the end of the stay is greater than the total number of days or when start day is negative"""


class NoneAlternativeError(Exception):
    """Exception raised when there is no alternative room available"""
