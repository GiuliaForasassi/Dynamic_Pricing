import random
from datetime import date, timedelta

import numpy as np
from loguru import logger
from scipy.stats import norm, rv_continuous

from pricing_advertising.exceptions.exceptions import InvalidReservationWindowError
from pricing_advertising.requests.request import Request
from pricing_advertising.utils.constants import (
    MAX_DAYS_STAY,
    MAX_GUESTS,
    MIN_GUESTS,
    MIN_NUM_REQUESTS_FACTOR,
    STD_FACTOR,
)


class RequestGenerator:
    def __init__(
        self, avg_num_requests: int, end_simulation: date, distribution: rv_continuous | None = None
    ) -> None:
        self.avg_num_requests = avg_num_requests
        self.end_simulation = end_simulation

        # If no distribution is provided, we use a normal distribution (loc: mean, scale: std)
        if distribution is None:
            self.distribution = norm(
                loc=self.avg_num_requests, scale=STD_FACTOR * self.avg_num_requests
            )
        else:
            self.distribution = distribution

    def generate_requests(self, today: date) -> np.ndarray:
        """
        Generate a list of requests for a given date.

        Args:
            today (date): The date for which to generate requests.

        Returns:
            np.ndarray[Request]: A list of generated requests.
        """
        num_requests = self._sample_num_requests()
        requests = []

        # generation of requests
        for _ in range(num_requests):
            request = self._generate_request(today)
            requests.append(request)
        return np.array(requests)

    def _sample_num_requests(self, size: int = 1) -> int:
        """
        Sample the number of requests based on a distribution and a minimum factor.

        This method samples the number of requests from a specified distribution
        and ensures that the sampled number is at least a minimum value, which is
        calculated as a factor of the average number of requests. The sampled number
        is then rounded to the nearest integer.

        Returns:
            int: The sampled number of requests, rounded to the nearest integer.
        """

        min_num_requests = MIN_NUM_REQUESTS_FACTOR * self.avg_num_requests

        # Sample the number of requests from the distribution provided
        num_requests = max(self.distribution.rvs(size=size), min_num_requests)

        # Round to the nearest integer
        num_requests = int(np.round(num_requests))
        return num_requests

    def _generate_request(self, today: date) -> Request:
        """
        Generates a reservation request for a given date.

        Args:
            today (date): The current date from which the reservation is generated.

        Returns:
            Request: A request object containing the reservation date, duration of stay, and number of guests.

        Raises:
            RequestGeneratorError: If there are no more days available for reservations.

        """
        remaining_days = (self.end_simulation - today).days
        if remaining_days <= 0:
            raise InvalidReservationWindowError(remaining_days=remaining_days)

        # reservation date is between today and the last day of the simulation
        t_reservation = today + timedelta(days=random.randint(1, remaining_days))
        days_stay = random.randint(1, min(MAX_DAYS_STAY, remaining_days))

        # number of guests can be 1, 2 or 3
        num_guests = random.randint(MIN_GUESTS, MAX_GUESTS)
        return Request(t_reservation, days_stay, num_guests)


if __name__ == "__main__":
    generator = RequestGenerator(avg_num_requests=10, end_simulation=date(2025, 12, 31))
    requests = generator.generate_requests(date.today())
    for request in requests:
        logger.info(request)
    logger.info(f"Number of requests: {len(requests)}")
