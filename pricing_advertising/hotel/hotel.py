from datetime import date, timedelta

import numpy as np

from pricing_advertising.clients.client import Client
from pricing_advertising.exception.booking import ClientRefusedError, RoomUnavailableError
from pricing_advertising.hotel.room import Room, RoomCategory
from pricing_advertising.policies.pricing.policy import PricingPolicy


class Hotel:
    def __init__(
        self,
        num_total_rooms: int,
        num_category1_rooms: int,
        num_category2_rooms: int,
        num_category3_rooms: int,
        start_simulation: date,
        end_simulation: date,
    ) -> None:
        self.num_total_rooms = num_total_rooms
        self.start_simulation = start_simulation  # day of start simulation
        self.end_simulation = end_simulation  # day of end simulation
        self.total_days = (end_simulation - start_simulation).days + 1

        self.num_rooms = {
            RoomCategory.CATEGORY1: num_category1_rooms,
            RoomCategory.CATEGORY2: num_category2_rooms,
            RoomCategory.CATEGORY3: num_category3_rooms,
        }

        self.rooms: list[Room] = []
        for category in RoomCategory:
            for _ in range(self.num_rooms[category]):
                room = Room(len(self.rooms), category)
                self.rooms.append(room)

        # matrix of shape (num_total_rooms, total_days) that indicates the availability of each room
        self.occupancy = np.zeros(
            (num_total_rooms, self.total_days), dtype=bool
        )  # initially all rooms are available

    # Check if the room is available for the client requested period
    def room_is_available(self, room: Room, client: Client) -> bool:
        room_idx = room.room_id
        reservation_idx = (client.t_reservation - self.start_simulation).days
        days_stay = client.days_stay
        # Ensure the requested period is within the simulation period
        if reservation_idx < 0 or reservation_idx + days_stay >= self.total_days:
            return False
        return np.all(self.occupancy[room_idx, reservation_idx : reservation_idx + days_stay] == 0)

    # REVIEW THIS PART, PROBABLY IT WILL MOVE TO THE ENVIRONMENT
    def handle_daily_requests(self, clients: list[Client], policy: PricingPolicy, today: date):
        # generate prices: policy + propose other categories rooms
        # prices is a matrix of shape (num_category, total_days)
        prices: np.ndarray = policy.select_price(self.rooms, clients)
        # TODO: consider past prices

        # step 1a
        # Matching clients with rooms
        # Assumption: matching is indipendent from the price
        assignements = {}
        for client in clients:
            if client.num_guests > 3:
                raise RoomUnavailableError("No room available for more than 3 guests")

            category = client.get_category()

            for room in self.rooms[category]:
                if self.room_is_available(room, client) and not self.room_already_assigned(
                    room, client, assignements
                ):
                    assignements[client] = room
        # step 1b
        # Propose other categories rooms
        for client in clients:
            if client not in assignements:
                category = client.get_category()
                for alternative_category in room.alternatives[category]:
                    for room in self.rooms[alternative_category]:
                        if self.room_is_available(room, client) and not self.room_already_assigned(
                            room, client, assignements
                        ):
                            assignements[client] = room

        # step 2
        # Notify clients: acceptance + update availability
        for client in clients:
            if client not in assignements:
                continue

            proposed_room: Room = assignements[client]
            t_advance_index = (client.t_reservation - today).days
            # Compute the total price for the client's stay
            proposed_price = prices[
                proposed_room.category.value - 1,
                t_advance_index : t_advance_index + client.days_stay,
            ].sum()

            if client.accept(proposed_price, proposed_room):
                index_reservation = self.date_to_index(client.t_reservation)
                # TODO: save somewhere the accepted price, so that we can use it for the constraints in the future
                self.occupancy[
                    proposed_room.room_id, index_reservation : index_reservation + client.days_stay
                ] = False
                return proposed_room
            else:
                raise ClientRefusedError("Client refused the price")

    # Check if the room is already assigned in the days specified by the client
    def room_already_assigned(
        self, room: Room, client: Client, assignements: dict[Client, Room]
    ) -> bool:
        for assigned_client, assigned_room in assignements.items():
            overlap = False
            if (
                client.t_reservation <= assigned_client.t_reservation
                and client.days_stay > (assigned_client.t_reservation - client.t_reservation).days
            ) or (
                client.t_reservation > assigned_client.t_reservation
                and assigned_client.t_reservation + timedelta(days=assigned_client.days_stay)
                > client.t_reservation
            ):
                overlap = True
            if room == assigned_room and overlap:
                return True
        return False

    # def get_occupancy(self, )
