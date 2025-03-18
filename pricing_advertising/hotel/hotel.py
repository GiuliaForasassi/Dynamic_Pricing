from loguru import logger

from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.models.models import NumHotelRooms


class Hotel:
    def __init__(
        self,
        num_category1_rooms: int,
        num_category2_rooms: int,
        num_category3_rooms: int,
    ) -> None:
        self.rooms = NumHotelRooms(
            num_category1_rooms=num_category1_rooms,
            num_category2_rooms=num_category2_rooms,
            num_category3_rooms=num_category3_rooms,
            num_total_rooms=num_category1_rooms + num_category2_rooms + num_category3_rooms,
        )

    def get_num_rooms_per_category(self, category: RoomCategory) -> int:
        """
        Get the number of rooms available for a specific category.

        Args:
            category (RoomCategory): The category of the room.

        Returns:
            int: The number of rooms available in the specified category.
        """
        if category == RoomCategory.CATEGORY1:
            return self.rooms.num_category1_rooms
        elif category == RoomCategory.CATEGORY2:
            return self.rooms.num_category2_rooms
        elif category == RoomCategory.CATEGORY3:
            return self.rooms.num_category3_rooms
        else:
            raise ValueError(f"Invalid room category: {category}")


if __name__ == "__main__":
    hotel = Hotel(num_category1_rooms=10, num_category2_rooms=20, num_category3_rooms=30)
    logger.info(
        f"Number of category 1 rooms: {hotel.get_num_rooms_per_category(RoomCategory.CATEGORY1)}"
    )
    logger.info(
        f"Number of category 2 rooms: {hotel.get_num_rooms_per_category(RoomCategory.CATEGORY2)}"
    )
    logger.info(
        f"Number of category 3 rooms: {hotel.get_num_rooms_per_category(RoomCategory.CATEGORY3)}"
    )
    logger.info(f"Total number of rooms: {hotel.rooms.num_total_rooms}")

    assert hotel.get_num_rooms_per_category(RoomCategory.CATEGORY1) == 10
    assert hotel.get_num_rooms_per_category(RoomCategory.CATEGORY2) == 20
    assert hotel.get_num_rooms_per_category(RoomCategory.CATEGORY3) == 30
    assert hotel.rooms.num_total_rooms == 60

    zero_hotel = Hotel(0, 0, 0)
    assert zero_hotel.rooms.num_total_rooms == 0, "Total rooms should be 0 when all inputs are 0"

    try:
        invalid_hotel = Hotel(-1, 10, 5)
    except ValueError:
        logger.info("Correctly captured negative room error.")
    try:
        hotel.get_num_rooms_per_category(None)  # type: ignore
    except ValueError as e:
        logger.info(f"Correctly captured invalid category error: {e}")

    logger.info("All tests passed successfully!")
