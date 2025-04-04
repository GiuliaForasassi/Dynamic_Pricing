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
    SEEDS,
    STD_FACTOR,
)


class RequestGenerator:
    def __init__(
        self,
        avg_num_requests: int,
        end_simulation: date,
        distribution: rv_continuous | None = None,
        seed: int | None = None,
    ) -> None:
        self.seed = seed

        self.np_rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)
        self.avg_num_requests = avg_num_requests
        self.end_simulation = end_simulation

        if not isinstance(end_simulation, date):
            raise ValueError("The end_simulation must be a date object")

        if avg_num_requests < 0:
            raise ValueError("The average number of requests must be greater than 0")

        # If no distribution is provided, we use a normal distribution (loc: mean, scale: std)
        if distribution is None:
            self.distribution = norm(
                loc=self.avg_num_requests, scale=STD_FACTOR * self.avg_num_requests
            )
        else:
            self.distribution = distribution

    def generate_requests(self, today: date) -> np.ndarray:
        """
        Generate an array of requests for each day from the given start date until the end of the simulation.

        Args:
            today (date): The starting date for generating requests.

        Returns:
            np.ndarray: A numpy array containing all the generated requests for the simulation period.
        """
        if not isinstance(today, date):
            raise ValueError("The start date must be a date object")

        if today >= self.end_simulation:
            raise ValueError("The start date must be before the end of the simulation")

        all_requests = []
        while today < self.end_simulation - timedelta(days=1):
            requests = self.generate_daily_requests(today)
            all_requests.extend(requests)
            today += timedelta(days=1)
        return np.array(all_requests)

    def generate_daily_requests(self, today: date) -> list[Request]:
        """
        Generate a list of requests for a given date.

        Args:
            today (date): The date for which to generate requests.

        Returns:
            np.ndarray[Request]: A list of generated requests.
        """
        num_requests = self._sample_num_requests()
        requests: list[Request] = []

        # generation of requests
        for _ in range(num_requests):
            request = self._generate_request(today)
            requests.append(request)
        return requests

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

        min_num_requests = max(1, MIN_NUM_REQUESTS_FACTOR * self.avg_num_requests)

        # Sample the number of requests from the distribution provided
        num_requests = max(
            self.distribution.rvs(size=size, random_state=self.seed), min_num_requests
        )

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
        if remaining_days <= 1:
            raise InvalidReservationWindowError(remaining_days=remaining_days)

        # reservation date is between today and the last day of the simulation
        # I can't reserve for today, so I add 1 to the randint
        # Subtract 1 because days stay must me at least 1
        t_reservation = today + timedelta(
            days=self.py_rng.randint(1, remaining_days - 1)
        )  # TODO: change this!
        remaining_days_from_reservation = (self.end_simulation - t_reservation).days
        days_stay = self.py_rng.randint(1, min(MAX_DAYS_STAY, remaining_days_from_reservation))

        # number of guests can be 1, 2 or 3
        num_guests = self.py_rng.randint(MIN_GUESTS, MAX_GUESTS)
        return Request(today, t_reservation, days_stay, num_guests)


if __name__ == "__main__":
    generator = RequestGenerator(
        seed=SEEDS[0], avg_num_requests=10, end_simulation=date(2025, 3, 31)
    )
    requests = generator.generate_requests(date.today())
    logger.info(f"Number of requests: {len(requests)}")
    logger.info(f"Shape: {requests.shape}")
    for request in requests:
        break
        logger.info(request)

    print(
        "-------------------------------------------------------------------------------------------------------------------------------------------------------------"
    )

    generator2 = RequestGenerator(
        avg_num_requests=10,
        end_simulation=date(2025, 3, 31),
        seed=SEEDS[0],
        distribution=norm(loc=10, scale=2),  # type: ignore
    )
    requests = generator2.generate_requests(date.today())
    logger.info(f"Number of requests: {len(requests)}")
    logger.info(f"Shape: {requests.shape}")
    for request in requests:
        logger.info(request)

    generator = RequestGenerator(avg_num_requests=10, end_simulation=date(2025, 3, 31))
