from loguru import logger

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

    def __str__(self) -> str:
        return f"Room {self.room_id} ({self.category.name})"


if __name__ == "__main__":
    room1 = Room(1, RoomCategory.CATEGORY1)
    logger.info(room1)

    assert room1.room_id == 1
    assert room1.category == RoomCategory.CATEGORY1

    room2 = Room(2, RoomCategory.CATEGORY2)
    logger.info(room2)

    assert room2.room_id == 2
    assert room2.category == RoomCategory.CATEGORY2

    assert ROOM_ALTERNATIVES[RoomCategory.CATEGORY1] == [
        RoomCategory.CATEGORY2,
        RoomCategory.CATEGORY3,
    ], "Incorrect alternatives for CATEGORY1"
    assert ROOM_ALTERNATIVES[RoomCategory.CATEGORY2] == [RoomCategory.CATEGORY3], (
        "Incorrect alternatives for CATEGORY2"
    )
    assert ROOM_ALTERNATIVES[RoomCategory.CATEGORY3] == [], "CATEGORY3 should have no alternatives"
    logger.info("All tests passed successfully!")
