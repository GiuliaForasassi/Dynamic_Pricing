from pricing_advertising.enums.enums import RoomCategory

# Alternative rooms in case the room is not available
ROOM_ALTERNATIVES = {
    RoomCategory.CATEGORY1: [RoomCategory.CATEGORY2, RoomCategory.CATEGORY3],
    RoomCategory.CATEGORY2: [RoomCategory.CATEGORY3],
    RoomCategory.CATEGORY3: [],
}


class Room:
    def __init__(self, room_id: int, category: RoomCategory) -> None:
        self.room_id = room_id  # id of the room
        self.category = category  # category of the room
