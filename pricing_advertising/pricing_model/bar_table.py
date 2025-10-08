from typing import Mapping, Sequence

from pricing_advertising.enums.enums import RoomCategory

BAR_TABLE: dict[RoomCategory, list[float]] = {
    RoomCategory.CATEGORY1: [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0],
    RoomCategory.CATEGORY2: [90.0, 96.0, 102.0, 108.0, 114.0, 120.0, 126.0, 132.0, 138.0, 144.0],
    RoomCategory.CATEGORY3: [120.0, 128.0, 136.0, 144.0, 152.0, 160.0, 168.0, 176.0, 184.0, 192.0],
}

# --- Check of consistency ---
_lengths = {len(v) for v in BAR_TABLE.values()}
if len(_lengths) != 1:
    raise ValueError(
        f"All room types must have the same number of BAR levels, but found different lengths: {_lengths}"
    )


# --- Helper functions to read-only access ---
def room_types() -> list[str]:
    """Returns the list of room types available in the bar table."""
    return [room.name for room in BAR_TABLE.keys()]


def n_levels() -> int:
    """Returns the number of pricing levels available for each room type."""
    return len(next(iter(BAR_TABLE.values())))


def get_price(room_type: str, level: int) -> float:
    """Returns the price for a given room type and pricing level.

    Args:
        room_type (str): The type of the room (e.g., "ROOM1", "ROOM2", "ROOM3").
        level (int): The pricing level (0-indexed).

    Returns:
        float: The price for the specified room type and pricing level.
    """
    prices = BAR_TABLE[RoomCategory[room_type]]
    bar_level = max(0, min(level, len(prices) - 1))  # Ensure level is within valid range
    return prices[bar_level]


def as_mapping() -> Mapping[str, Sequence[float]]:
    """Returns the BAR_TABLE as a mapping."""
    return {room.name: prices for room, prices in BAR_TABLE.items()}


if __name__ == "__main__":
    # Stampa la tabella BAR in formato leggibile
    print("=== TABELLAAAA BAR ===")
    header = "BAR_LEVEL".ljust(10) + "  " + "  ".join(room.center(10) for room in room_types())
    print(header)
    print("-" * len(header))
    for level in range(n_levels()):
        row_values = [f"{get_price(room, level):>10.2f}" for room in room_types()]
        print(f"{str(level).ljust(10)}  " + "  ".join(row_values))
