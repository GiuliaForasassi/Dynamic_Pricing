class ClientRefusedError(Exception):
    """Exception raised when the client refuses the offer"""


class RoomUnavailableError(Exception):
    """Exception raised when no room is available for requested category"""
