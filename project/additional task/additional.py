import numpy as np
import cv2

class Coordinates:
    def __init__(self, roll, pitch, yaw, initial_position=(0, 0, 0)):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.initial_position = initial_position

    def __repr__(self):
        trajectory = self._trajectory(time_step=0.01, total_time=2)
        return f'{trajectory}'

    def __str__(self):
        trajectory = self._trajectory(time_step=0.01, total_time=2)
        return f"{trajectory}"

    def __iter__(self):
        self._index = 0
        self._positions = self._trajectory(time_step=0.01, total_time=2)
        return self

    def __next__(self):
        if self._index < len(self._positions):
            result = self._positions[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration

    def _roll_matrix(self):
        roll = np.radians(self.roll)
        _R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        return _R_roll

    def _pitch_matrix(self):
        pitch = np.radians(self.pitch)
        _R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        return _R_pitch

    def _yaw_matrix(self):
        yaw = np.radians(self.yaw)
        _R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        return _R_yaw

    def _rotation_matrix(self):
        return np.dot(self._yaw_matrix(), np.dot(self._pitch_matrix(), self._roll_matrix()))

    def _vector_speed_global(self):
        v_local = np.array([1, 0, 0])
        return np.dot(self._rotation_matrix(), v_local)

    def _wind_global(self):
        wind_local = np.array([0, 0.1, 0])
        return np.dot(self._rotation_matrix(), wind_local)

    def _initial_velocity(self):
        return self._vector_speed_global() + self._wind_global()

    def _trajectory(self, time_step=0.01, total_time=2):
        g = 9.81
        positions = []
        position = np.array(self.initial_position, dtype=float)
        velocity = np.array(self._initial_velocity())
        for _ in np.arange(0, total_time, time_step):
            positions.append(position.copy())
            velocity[2] -= g * time_step
            position += velocity * time_step
            if position[2] < 0:
                position[2] = 0
                positions.append(position.copy())
                break
        return np.array(positions)

def draw_trajectory_on_image(image_path, coordinates, output_path):
    """
    Load the image
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load")
    """
    Calculate the trajectory
    """
    trajectory = coordinates._trajectory()
    """
    Normalize function to convert coordinates to image pixels
    """
    def normalize(coord, max_value, size):
        return int((coord + max_value / 2) * size / max_value)

    """
    Get image dimensions
    """
    height, width, _ = image.shape
    """
    Determine the max values for normalization
    """
    max_x = max(abs(trajectory[:, 0].min()), abs(trajectory[:, 0].max()))
    max_y = max(abs(trajectory[:, 1].min()), abs(trajectory[:, 1].max()))
    max_z = max(abs(trajectory[:, 2].min()), abs(trajectory[:, 2].max()))
    """
    Draw the trajectory on the image
    """
    for pos in trajectory:
        x, y, z = pos
        cv2.circle(image, (normalize(x, max_x, width), normalize(y, max_y, height)), 2, (0, 0, 255), -1)
    """
    Save the output image
    """
    cv2.imwrite(output_path, image)

# Example usage
coordinates = Coordinates(1000, 2000, 3000, initial_position=(0, 0, 1))
draw_trajectory_on_image('image.png', coordinates, 'output_image.png')
