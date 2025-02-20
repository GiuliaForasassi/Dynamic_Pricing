from enum import Enum


class RoomCategory(Enum):
    CATEGORY1 = 1  # single room (1 guest)
    CATEGORY2 = 2  # double room (2 guests)
    CATEGORY3 = 3  # family room (3 guests)


class Room:
    def __init__(self, room_id: int, category: RoomCategory) -> None:
        self.room_id = room_id  # id of the room
        self.category = category  # category of the room

        # Alternative rooms in case the room is not available
        self.alternatives = {
            RoomCategory.CATEGORY1: [RoomCategory.CATEGORY2, RoomCategory.CATEGORY3],
            RoomCategory.CATEGORY2: [RoomCategory.CATEGORY3],
            RoomCategory.CATEGORY3: [],
        }
