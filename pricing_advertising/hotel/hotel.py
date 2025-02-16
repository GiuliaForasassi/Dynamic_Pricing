from datetime import date, timedelta
from random import random

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
    ) -> None:
        # Initialize the hotel
        self.num_total_rooms = num_total_rooms
        self.num_category1_rooms = num_category1_rooms
        self.num_category2_rooms = num_category2_rooms
        self.num_category3_rooms = num_category3_rooms
        self.rooms: dict[RoomCategory, list[Room]] = {
            RoomCategory.CATEGORY1: [],
            RoomCategory.CATEGORY2: [],
            RoomCategory.CATEGORY3: [],
        }

    def add_room(self, room: Room) -> None:
        self.rooms[room.category].append(room)

    @staticmethod
    def create_hotel(
        num_total_rooms: int, num_category1: int, num_category2: int, num_category3: int
    ) -> "Hotel":
        # Validate input parameters
        if num_total_rooms != num_category1 + num_category2 + num_category3:
            raise ValueError(
                "The total number of rooms must be equal to the sum of the rooms in each category"
            )
        # Create the hotel
        hotel = Hotel(num_total_rooms, num_category1, num_category2, num_category3)
        # Populate the hotel with rooms
        for i in range(num_total_rooms):
            if i < num_category1:
                category = RoomCategory.CATEGORY1
            elif i < num_category1 + num_category2:
                category = RoomCategory.CATEGORY2
            else:
                category = RoomCategory.CATEGORY3
            room = Room(f"room{i}", category)
            hotel.add_room(room)
        return hotel

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
                if room.is_available_days(
                    client.t_reservation, client.days_stay
                ) and not self.room_already_assigned(room, client, assignements):
                    assignements[client] = room
        # step 1b
        # Propose other categories rooms
        for client in clients:
            if client not in assignements:
                category = client.get_category()
                for alternative_category in room.alternatives[category]:
                    for room in self.rooms[alternative_category]:
                        if room.is_available_days(
                            client.t_reservation, client.days_stay
                        ) and not self.room_already_assigned(room, client, assignements):
                            assignements[client] = room

        # step 2
        # Notify clients: acceptance + update availability
        for client in clients:
            if client not in assignements:
                continue
            proposed_room: Room = assignements[client]
            t_advance_index = (client.t_reservation - today).days
            proposed_price = prices[
                proposed_room.category.value - 1,
                t_advance_index : t_advance_index + client.days_stay,
            ].sum()
            if client.accept(proposed_price, room):
                # TODO: save somewhere the accepted price, so that we can use it for the constraints in the future
                for i in range(client.days_stay):
                    room.availability[client.t_reservation + timedelta(days=i)] = False
                return room
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
