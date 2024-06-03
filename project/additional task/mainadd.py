from additional import Coordinates, draw_trajectory_on_image
roll = int(input("Enter roll:"))
pitch = int(input("Enter pitch:"))
yaw = int(input("Enter yaw:"))
initial_position_str = input("Enter initial_position:")
try:
    initial_position = tuple(map(int, initial_position_str.split(',')))
except ValueError:
    print("Invalid input. Please enter a valid tuple of integers separated by commas.")

object = Coordinates(roll,pitch,yaw,initial_position)
draw_trajectory_on_image('image.png',object,'output_image.png')s
