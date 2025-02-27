from datetime import timedelta

import numpy as np
from pydantic import BaseModel, Field

from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.exceptions.exceptions import (
    NotValidPeriodError,
)
from pricing_advertising.requests.request import Request
from pricing_advertising.utils.constants import NUM_CATEGORIES


class NumHotelRooms(BaseModel):
    """Number of rooms per category"""

    num_category1_rooms: int = Field(description="Number of category 1 rooms")
    num_category2_rooms: int = Field(description="Number of category 2 rooms")
    num_category3_rooms: int = Field(description="Number of category 3 rooms")
    num_total_rooms: int = Field(description="Total number of rooms")


class Hotel:
    def __init__(
        self,
        total_days: int,
        num_category1_rooms: int,
        num_category2_rooms: int,
        num_category3_rooms: int,
    ) -> None:
        self.total_days = total_days  # total number of simulation days

        self.rooms = NumHotelRooms(
            num_category1_rooms=num_category1_rooms,
            num_category2_rooms=num_category2_rooms,
            num_category3_rooms=num_category3_rooms,
            num_total_rooms=num_category1_rooms + num_category2_rooms + num_category3_rooms,
        )

        # matrix of shape (num_categories, total_days) to keep track of the occupancy
        self.occupancy = np.zeros(
            (NUM_CATEGORIES, total_days), dtype=int
        )  # occupancy matrix: number of rooms occupied per category per day

    def check_availability(self, request: Request) -> bool:
        """
        Checks the availability of rooms for a given request.

        Args:
            request (Request): The request object containing the reservation details.

        Returns:
            bool: True if there is availability for the requested category and period, False otherwise.

        Raises:
            NotValidPeriodError: If the requested period is not valid.
        """
        category = self._category
        start_day = request.t_reservation
        days_stay = request.days_stay
        end_day = start_day + days_stay

        # Check if the period is valid
        if start_day < 0 or start_day + days_stay > self.total_days:
            raise NotValidPeriodError()

        # Check if there is a room available for the requested category and period
        total_rooms_for_category = self.get_num_rooms_per_category(category)
        category_index = category.value

        return np.all(self.occupancy[category_index, start_day:end_day] < total_rooms_for_category)

    def get_num_rooms_per_category(self, category: RoomCategory) -> int:
        """
        Get the number of rooms available for a specific category.

        Args:
            category (RoomCategory): The category of the room.

        Returns:
            int: The number of rooms available in the specified category.
        """
        return self.rooms[category]

    @property
    def category(self) -> RoomCategory:
        """Return the room category based on the number of guests"""
        self._category = RoomCategory(self.num_guests)
        return self._category
