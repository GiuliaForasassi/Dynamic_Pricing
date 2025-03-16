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
        return getattr(self.rooms, f"num_{category.value}_rooms")
