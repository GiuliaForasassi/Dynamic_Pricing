class Room:  # noqa: INP001
    def __init__(self, room_id: int, category: int):
        self.room_id = room_id  # id of the room
        self.category = category  # category of the room
        self.occupancy = {}  # dictionary with the occupancy of the room for each day
