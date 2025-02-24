import random
from datetime import date, timedelta

from scipy.stats import rv_continuous

from pricing_advertising.exception.booking import RequestGeneratorError
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
        self, avg_num_requests: int, end_simulation: date, distribution: rv_continuous
    ) -> None:
        self.avg_num_requests = avg_num_requests
        self.end_simulation = end_simulation
        self.distribution = distribution

    # Generate a list of requests
    def generate_requests(self, today: date) -> list[Request]:
        # number of requests is sampled from a normal distribution
        num_requests = self.sample_num_requests()
        # initialize the list of requests
        requests = []
        # generation of requests
        for _ in range(num_requests):
            request = self.generate_request(today)
            requests.append(request)
        return requests

    # Sample the number of requests from a normal distribution
    def sample_num_requests(self) -> int:
        # standard deviation is 10% of the average number of requests
        std = STD_FACTOR * self.avg_num_requests
        # minimum number of requests is 10% of the average number of requests
        min_num_requests = MIN_NUM_REQUESTS_FACTOR * self.avg_num_requests
        num_requests = max(random.gauss(self.avg_num_requests, std), min_num_requests)
        # round the number of requests to the nearest integer
        num_requests = round(num_requests)
        return num_requests

    # Generate a single request
    def generate_request(self, today: date) -> Request:
        remaining_days = (self.end_simulation - today).days
        if remaining_days <= 0:
            raise RequestGeneratorError("No more days available for reservations")
        # reservation date is between today and the last day of the simulation
        t_reservation = today + timedelta(days=random.randint(1, remaining_days))
        # number of days the request will stay is between 1 and 10
        days_stay = random.randint(1, min(MAX_DAYS_STAY, remaining_days))
        # number of guests can be 1, 2 or 3
        num_guests = random.randint(MIN_GUESTS, MAX_GUESTS)
        return Request(t_reservation, days_stay, num_guests)
