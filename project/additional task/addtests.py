import unittest
import numpy as np
import cv2
import os

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
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load")

    trajectory = coordinates._trajectory()

    def normalize(coord, max_value, size):
        return int((coord + max_value / 2) * size / max_value)

    height, width, _ = image.shape

    max_x = max(abs(trajectory[:, 0].min()), abs(trajectory[:, 0].max()))
    max_y = max(abs(trajectory[:, 1].min()), abs(trajectory[:, 1].max()))
    max_z = max(abs(trajectory[:, 2].min()), abs(trajectory[:, 2].max()))

    for pos in trajectory:
        x, y, z = pos
        cv2.circle(image, (normalize(x, max_x, width), normalize(y, max_y, height)), 2, (0, 0, 255), -1)

    cv2.imwrite(output_path, image)

class TestCoordinates(unittest.TestCase):
    def test_roll_matrix(self):
        coord = Coordinates(90, 0, 0)
        expected_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        np.testing.assert_array_almost_equal(coord._roll_matrix(), expected_matrix)

    def test_pitch_matrix(self):
        coord = Coordinates(0, 90, 0)
        expected_matrix = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
        np.testing.assert_array_almost_equal(coord._pitch_matrix(), expected_matrix)

    def test_yaw_matrix(self):
        coord = Coordinates(0, 0, 90)
        expected_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(coord._yaw_matrix(), expected_matrix)

    def test_rotation_matrix(self):
        coord = Coordinates(90, 90, 0)
        expected_matrix = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
        np.testing.assert_array_almost_equal(coord._rotation_matrix(), expected_matrix)

    def test_vector_speed_global(self):
        coord = Coordinates(0, 0, 0)
        expected_vector = np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(coord._vector_speed_global(), expected_vector)

    def test_wind_global(self):
        coord = Coordinates(0, 0, 0)
        expected_wind = np.array([0, 0.1, 0])
        np.testing.assert_array_almost_equal(coord._wind_global(), expected_wind)

    def test_initial_velocity(self):
        coord = Coordinates(0, 0, 0)
        expected_velocity = np.array([1, 0.1, 0])
        np.testing.assert_array_almost_equal(coord._initial_velocity(), expected_velocity)

    def test_trajectory(self):
        coord = Coordinates(0, 0, 0, initial_position=(0, 0, 100))
        trajectory = coord._trajectory(time_step=0.01, total_time=2)
        self.assertGreater(len(trajectory), 0)
        self.assertEqual(trajectory[-1][2], 0)

class TestDrawTrajectoryOnImage(unittest.TestCase):
    def test_draw_trajectory_on_image(self):
        coord = Coordinates(10, 20, 30, initial_position=(0, 0, 100))
        input_image_path = 'input_image.jpg'
        output_image_path = 'output_image.jpg'
        # Create a blank input image
        blank_image = np.zeros((500, 500, 3), np.uint8)
        cv2.imwrite(input_image_path, blank_image)
        # Draw trajectory on image
        draw_trajectory_on_image(input_image_path, coord, output_image_path)
        # Check if output image is created
        self.assertTrue(os.path.exists(output_image_path))
        # Cleanup
        os.remove(input_image_path)
        os.remove(output_image_path)

if __name__ == '__main__':
    unittest.main()

