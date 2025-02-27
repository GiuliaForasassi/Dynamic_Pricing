class BaseBookingError(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


class BaseHotelError(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


class BaseRequestError(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


class BaseImplementationError(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)


class BaseBaseEnvError(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)
