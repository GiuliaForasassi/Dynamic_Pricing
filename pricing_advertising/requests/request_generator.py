import random
from datetime import date, timedelta

from pricing_advertising.requests.request import Request


class RequestGenerator:
    def __init__(self, avg_num_requests: int, end_simulation: date) -> None:
        self.avg_num_requests = avg_num_requests
        self.end_simulation = end_simulation

    # Generate a list of clients
    def generate_requests(self, today: date) -> list[Request]:
        # number of clients is sampled from a normal distribution
        num_requests = self.sample_num_requests()
        # initialize the list of clients
        requests = []
        # generation of clients
        for _ in range(num_requests):
            request = self.generate_request(today)
            requests.append(request)

        return requests

    # Sample the number of clients from a normal distribution
    def sample_num_requests(self) -> int:
        # standard deviation is 10% of the average number of clients
        std = 0.1 * self.avg_num_requests
        # minimum number of clients is 10% of the average number of clients
        min_num_requests = 0.2 * self.avg_num_requests
        num_requests = max(random.gauss(self.avg_num_requests, std), min_num_requests)
        # round the number of clients to the nearest integer
        num_requests = round(num_requests)
        return num_requests

    # Generate a single client
    def generate_request(self, today: date) -> Request:
        remaining_days = (self.end_simulation - today).days
        assert remaining_days > 0, "No more days available for reservations"
        # reservation date is between today and the last day of the simulation
        t_reservation = today + timedelta(days=random.randint(1, remaining_days))
        # number of days the client will stay is between 1 and 10
        days_stay = random.randint(1, min(10, remaining_days))
        # number of guests can be 1, 2 or 3
        num_guests = random.randint(1, 3)
        return Request(t_reservation, days_stay, num_guests)
