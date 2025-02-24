class ClientRefusedError(Exception):
    """Exception raised when the client refuses the offer"""


class RequestGeneratorError(Exception):
    """Exception raised when there is a problem with the RequestGenerator"""


class RoomUnavailableError(Exception):
    """Exception raised when no room is available for requested category"""
