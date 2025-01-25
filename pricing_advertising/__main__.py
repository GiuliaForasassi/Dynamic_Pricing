import random

from pricing_advertising.hotel.hotel import Hotel

if __name__ == '__main__':
    # Create a hotel with 30 rooms
    hotel = Hotel(total_rooms=30)

    # Adding rooms
    for r in range(1, hotel.total_rooms + 1):
        hotel.add_room(
            room_id=r, category=random.randint(1, 3), available=True
        )

    # Display all rooms
    print('All rooms in the hotel:')
    for room in hotel.rooms:
        print(room)

    # Book a room given the category
    booked_room = hotel.book_room(category=2)
    if booked_room:
        print(booked_room)
    else:
        print('No rooms available in the specified category.')

    # Display all rooms after booking
    for room in hotel.get_available_rooms_per_category(category=2):
        print(room)
