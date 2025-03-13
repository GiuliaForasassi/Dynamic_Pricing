from datetime import date, timedelta

import numpy as np

from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.exceptions.exceptions import (
    NotValidPeriodError,
    RoomUnavailableError,
)
from pricing_advertising.models.models import NumHotelRooms
from pricing_advertising.requests.request import Request
from pricing_advertising.utils.constants import NUM_CATEGORIES


class Hotel:
    def __init__(
        self,
        total_days: int,
        start_simulation: date,
        end_simulation: date,
        num_category1_rooms: int,
        num_category2_rooms: int,
        num_category3_rooms: int,
    ) -> None:
        self.total_days = total_days  # total number of simulation days
        self.start_simulation = start_simulation
        self.end_simulation = end_simulation

        self.rooms = NumHotelRooms(
            num_category1_rooms=num_category1_rooms,
            num_category2_rooms=num_category2_rooms,
            num_category3_rooms=num_category3_rooms,
            num_total_rooms=num_category1_rooms + num_category2_rooms + num_category3_rooms,
        )

        # matrix of shape (num_categories, total_days) to keep track of the occupancy
        self.occupancy = np.zeros((NUM_CATEGORIES, total_days), dtype=int)
        self.demand = np.zeros((NUM_CATEGORIES, total_days), dtype=int)

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
        # Get the category and period
        category = request.get_category()
        start_day = request.t_reservation
        days_stay = request.days_stay
        end_day = start_day + timedelta(days=days_stay)
        total_rooms_for_category = self.get_num_rooms_per_category(category)

        # Get the corresppnding indexes
        category_idx = category.value
        start_idx = (start_day - self.start_simulation).days
        end_idx = (end_day - self.start_simulation).days

        # Check if the period is valid
        if start_day < self.start_simulation or end_day >= self.end_simulation:
            raise NotValidPeriodError(
                start_period=start_day,
                end_period=end_day,
                start_simulation=self.start_simulation,
                end_simulation=self.end_simulation,
            )

        return bool(
            np.all(self.occupancy[category_idx, start_idx:end_idx] < total_rooms_for_category)
        )

    def get_num_rooms_per_category(self, category: RoomCategory) -> int:
        """
        Get the number of rooms available for a specific category.

        Args:
            category (RoomCategory): The category of the room.

        Returns:
            int: The number of rooms available in the specified category.
        """
        return getattr(self.rooms, f"num_{category.value}_rooms")

    def update_occupancy(self, request: Request) -> None:
        """
        Updates the occupancy matrix for a specified category and period based on the given request.

        Args:
            request (Request): The request object containing reservation details.

        Returns:
            None
        """
        # Get the category and period
        category = request.get_category()
        start_day = request.t_reservation
        days_stay = request.days_stay
        end_day = start_day + timedelta(days=days_stay)

        # Get the corresponding indexes
        category_idx = category.value
        start_idx = (start_day - self.start_simulation).days
        end_idx = (end_day - self.start_simulation).days

        total_rooms_for_category = self.get_num_rooms_per_category(category)

        # Check if the request period is valid and the category index is within bounds
        if days_stay <= 0:
            raise ValueError(
                "The number of days for the stay must be greater than 0. Received days_stay: {days_stay}"
            )
        if category_idx < 0 or category_idx >= self.occupancy.shape[0]:
            raise IndexError("Invalid category index.")
        if start_day < self.start_simulation or end_day > self.end_simulation:
            raise NotValidPeriodError(
                start_period=start_day,
                end_period=end_day,
                start_simulation=self.start_simulation,
                end_simulation=self.end_simulation,
            )

        # Check capacity for each day of the period
        if np.any(self.occupancy[category_idx, start_idx:end_idx] >= total_rooms_for_category):
            raise RoomUnavailableError(category=category_idx)

        # Update the occupancy matrix for the specified category and period
        self.occupancy[category_idx, start_idx:end_idx] += 1

    def update_demand(self, requests: list[Request], current_day: date) -> None:
        """
        Updates the demand for hotel categories based on the given requests for the previous day.

        Args:
            requests (list[Request]): A list of Request objects representing the requests made.
            current_day (date): The current day of the simulation.

        Raises:
            IndexError: If the previous day's index is out of bounds.

        Returns:
            None
        """
        previous_day = current_day - timedelta(days=1)
        previous_day_idx = (previous_day - self.start_simulation).days

        if previous_day_idx < 0 or previous_day_idx >= self.total_days:
            raise IndexError("Invalid index for previous day.")  # no updates it out of bounds

        category_counts = np.zeros(NUM_CATEGORIES, dtype=int)
        for request in requests:
            category = request.get_category()
            category_idx = category.value
            category_counts[category_idx] += 1

        self.demand[:, previous_day_idx] = category_counts
