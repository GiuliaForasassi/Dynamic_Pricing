import random
from datetime import date, timedelta

from pricing_advertising.clients.client import Client


class ClientGenerator:
    def __init__(self, avg_num_clients: int, end_simulation: date) -> None:
        self.avg_num_clients = avg_num_clients
        self.end_simulation = end_simulation

    # Generate a list of clients
    def generate_clients(self, today: date) -> list[Client]:
        # number of clients is sampled from a normal distribution
        num_clients = self.sample_num_clients()
        # initialize the list of clients
        clients = []
        # generation of clients
        for _ in range(num_clients):
            client = self.generate_client(today)
            clients.append(client)

        return clients

    # Sample the number of clients from a normal distribution
    def sample_num_clients(self) -> int:
        # standard deviation is 10% of the average number of clients
        std = 0.1 * self.avg_num_clients
        # minimum number of clients is 10% of the average number of clients
        min_num_clients = 0.2 * self.avg_num_clients
        num_clients = max(random.gauss(self.avg_num_clients, std), min_num_clients)
        # round the number of clients to the nearest integer
        num_clients = round(num_clients)
        return num_clients

    # Generate a single client
    def generate_client(self, today: date) -> Client:
        remaining_days = (self.end_simulation - today).days
        assert remaining_days > 0, "No more days available for reservations"
        # reservation date is between today and the last day of the simulation
        t_reservation = today + timedelta(days=random.randint(1, remaining_days))
        # number of days the client will stay is between 1 and 10
        days_stay = random.randint(1, min(10, remaining_days))
        # number of guests can be 1, 2 or 3
        num_guests = random.randint(1, 3)
        return Client(t_reservation, days_stay, num_guests)
