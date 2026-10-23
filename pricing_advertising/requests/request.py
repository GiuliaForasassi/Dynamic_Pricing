from dataclasses import dataclass
from datetime import date

from loguru import logger

from pricing_advertising.enums.enums import RoomCategory

# ------------------ OFFER TYPES ------------------
# "original" is the offer matching the request; the others are the alternatives the env can propose.
ORIGINAL_OFFER = "original"
CATEGORY_ALTERNATIVES = ("cat_1_to_2", "cat_1_to_3", "cat_2_to_3")
PERIOD_ALTERNATIVES = (
    "period_start_later_shorter",  # x+1, d-1
    "period_end_earlier",  # x, d-1
    "period_shift_earlier",  # x-1, d
    "period_shift_later",  # x+1, d
)
OFFER_TYPES = (ORIGINAL_OFFER, *CATEGORY_ALTERNATIVES, *PERIOD_ALTERNATIVES)
ALTERNATIVE_TYPES = OFFER_TYPES[1:]

_PERIOD_ALTERNATIVE_BY_DELTA = {
    (1, -1): "period_start_later_shorter",
    (0, -1): "period_end_earlier",
    (-1, 0): "period_shift_earlier",
    (1, 0): "period_shift_later",
}


# ------------------ USER REQUEST ------------------
class Request:
    def __init__(
        self, t_request: date, t_reservation: date, days_stay: int, num_guests: int
    ) -> None:
        if t_request > t_reservation:
            raise ValueError(
                f"The request date {t_request} must be before or equal the reservation date {t_reservation}."
            )
        self.t_request = t_request  # The date when the request was made
        self.t_reservation = t_reservation  # The date when the reservation is scheduled
        self.days_stay = days_stay  # The number of days for the stay
        self.num_guests = num_guests  # The number of guests for the reservation

    def __str__(self) -> str:
        return f"Request: {self.t_request} | Reservation: {self.t_reservation} | Stay: {self.days_stay} days | Guests: {self.num_guests}"

    # Get the total number of reservation days for the request
    @property
    def total_reservation_days(self) -> int:
        return self.days_stay + 1  # +1 to include the t_res day

    # Get the room category based on the number of guests
    @property
    def category(self) -> RoomCategory:
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

    # Set the room category based on the number of guests
    @category.setter
    def category(self, value: RoomCategory) -> None:
        if not isinstance(value, RoomCategory):
            raise ValueError("Invalid room category.")
        if value == RoomCategory.CATEGORY1:
            self.num_guests = 1
        elif value == RoomCategory.CATEGORY2:
            self.num_guests = 2
        elif value == RoomCategory.CATEGORY3:
            self.num_guests = 3
        else:
            raise ValueError(f"Unsupported RoomCategory: {value}.")

    # Get the number of days in advance the reservation was made
    @property
    def advance(self) -> int:
        return (self.t_reservation - self.t_request).days


# ------------------ HOTEL OFFER ------------------
@dataclass(frozen=True)
class HotelOffer:
    request: Request
    category: RoomCategory
    t_reservation: date
    days_stay: int

    # Check if the reservation period has changed compared to the original request
    @property
    def period_changed(self) -> bool:
        return (
            self.t_reservation != self.request.t_reservation
            or self.days_stay != self.request.days_stay
        )

    # Classify the offer with respect to the original request (one of OFFER_TYPES)
    @property
    def offer_type(self) -> str:
        if self.category != self.request.category:
            if self.period_changed:
                raise ValueError("Offers changing both category and period are not supported.")
            return f"cat_{self.request.category.value}_to_{self.category.value}"
        if not self.period_changed:
            return ORIGINAL_OFFER
        delta = (
            (self.t_reservation - self.request.t_reservation).days,
            self.days_stay - self.request.days_stay,
        )
        return _PERIOD_ALTERNATIVE_BY_DELTA[delta]
