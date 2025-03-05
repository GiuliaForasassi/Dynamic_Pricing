from datetime import date

from pydantic import BaseModel


class ObservationModel(BaseModel):
    """Base observation model."""

    occupancy: list[list[int]]
    demand: list[int]
    today: date
