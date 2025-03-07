from datetime import date

from pydantic import BaseModel, Field


class NumHotelRooms(BaseModel):
    """Number of rooms per category"""

    num_category1_rooms: int = Field(description="Number of category 1 rooms")
    num_category2_rooms: int = Field(description="Number of category 2 rooms")
    num_category3_rooms: int = Field(description="Number of category 3 rooms")
    num_total_rooms: int = Field(description="Total number of rooms")


class ObservationModel(BaseModel):
    """Base observation model."""

    occupancy: list[list[int]] = Field(
        default_factory=list, description="Occupancy matrix rappresenting rooms usage."
    )
    demand: list[int] = Field(
        default_factory=list, description="A list representing demand levels."
    )
    today: date = Field(description="The current date of the observation.")
