from datetime import timedelta

import numpy as np

from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.exceptions.booking import (
    NotValidPeriodError,
)
from pricing_advertising.requests.request import Request
from pricing_advertising.utils.constants import NUM_CATEGORIES


class Hotel:
    def __init__(
        self,
        total_days: int,
        num_category1_rooms: int,
        num_category2_rooms: int,
        num_category3_rooms: int,
    ) -> None:
        self.total_days = total_days  # total number of simulation days
        self.num_category1_rooms = num_category1_rooms
        self.num_category2_rooms = num_category2_rooms
        self.num_category3_rooms = num_category3_rooms

        self.num_total_rooms = num_category1_rooms + num_category2_rooms + num_category3_rooms

        self.num_rooms = {
            RoomCategory.CATEGORY1: num_category1_rooms,
            RoomCategory.CATEGORY2: num_category2_rooms,
            RoomCategory.CATEGORY3: num_category3_rooms,
        }

        # matrix of shape (num_categories, total_days) to keep track of the occupancy
        self.occupancy = np.zeros(
            (NUM_CATEGORIES, self.total_days), dtype=int
        )  # occupancy matrix: number of rooms occupied per category per day

    def check_availability(self, request: Request) -> bool:
        category = request.get_category()
        start_day = request.get_t_reservation()
        days_stay = request.get_days_stay()
        end_day = start_day + timedelta(days=days_stay)

        # Check if the period is valid
        if start_day < 0 or start_day + days_stay > self.total_days:
            raise NotValidPeriodError("The period is not valid.")

        # Check if there is a room available for the requested category and period
        total_rooms_for_category = self.get_num_rooms_per_category(self, category)
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
        return self.num_rooms[category]

    def get_total_days(self) -> int:
        """
        Calculate and return the total number of days.

        Returns:
            int: The total number of days.
        """
        return self.total_days

    def get_occupancy(self) -> np.ndarray:
        """
        Returns the occupancy data of the hotel.

        Returns:
            np.ndarray: An array representing the occupancy data.
        """
        return self.occupancy

    def get_num_total_rooms(self) -> int:
        """
        Returns the total number of rooms in the hotel.

        Returns:
            int: The total number of rooms.
        """
        return self.num_total_rooms
