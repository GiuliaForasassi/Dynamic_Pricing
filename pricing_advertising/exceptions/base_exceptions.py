class BaseBookingError(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


class BaseHotelError(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


class BaseRequestError(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


class BaseEnvError(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)
