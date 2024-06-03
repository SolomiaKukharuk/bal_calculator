import numpy as np
import tkinter as tk

class _Coordinates:
    def __init__(self, roll, pitch, yaw, initial_position=(0, 0, 0)):
        """
        Initializes the _Coordinates class with roll, pitch, yaw, and initial position.
        """
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.initial_position = initial_position

    def __repr__(self):
        """
        Returns a string representation of the trajectory.
        """
        trajectory = self._trajectory(time_step=0.01, total_time=2)
        return f'{trajectory}'

    def __str__(self):
        """
        Returns a string representation of the trajectory.
        """
        trajectory = self._trajectory(time_step=0.01, total_time=2)
        return f"{trajectory}"

    def __iter__(self):
        """
        Implements the iterator protocol.
        """
        self._index = 0
        self._positions = self._trajectory(time_step=0.01, total_time=2)  # Calculate the trajectory once
        return self

    def __next__(self):
        """
        Returns the next position in the trajectory.
        """
        if self._index < len(self._positions):
            result = self._positions[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration

    def _roll_matrix(self):
        """
        Returns the roll rotation matrix.
        """
        roll = np.radians(self.roll)
        _R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        return _R_roll

    def _pitch_matrix(self):
        """
        Returns the pitch rotation matrix.
        """
        pitch = np.radians(self.pitch)
        _R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        return _R_pitch

    def _yaw_matrix(self):
        """
        Returns the yaw rotation matrix.
        """
        yaw = np.radians(self.yaw)
        _R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        return _R_yaw

    def _rotation_matrix(self):
        """
        Returns the combined rotation matrix from yaw, pitch, and roll.
        """
        return np.dot(self._yaw_matrix(), np.dot(self._pitch_matrix(), self._roll_matrix()))

    def _vector_speed_global(self):
        """
        Returns the initial velocity vector in the global coordinate system.
        """
        v_local = np.array([1, 0, 0])
        return np.dot(self._rotation_matrix(), v_local)

    def _wind_global(self):
        """
        Returns the wind effect vector in the global coordinate system.
        """
        wind_local = np.array([0, 0.1, 0])
        return np.dot(self._rotation_matrix(), wind_local)

    def _initial_velocity(self):
        """
        Returns the initial velocity considering the wind effect.
        """
        return self._vector_speed_global() + self._wind_global()

    def _trajectory(self, time_step=0.01, total_time=2):
        """
        Calculates the trajectory of the projectile.
        """
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


class Draw(_Coordinates):
    def __init__(self, roll, pitch, yaw, initial_position=(0, 0, 0)):
        """
        Initializes the Draw class and calculates the trajectory.
        """
        super().__init__(roll, pitch, yaw, initial_position)
        self._positions = self._trajectory(time_step=0.01, total_time=2)

    def __str__(self):
        """
        Returns a string representation of the drawn trajectory.
        """
        return self.draw_trajectory()

    def __repr__(self):
        """
        Returns a string representation of the drawn trajectory.
        """
        return self.draw_trajectory()

    def get_canvas_properties(self):
        """
        Gets canvas properties from user input.
        """
        width = int(input("Enter the width for the canvases: "))
        height = int(input("Enter the height for the canvases: "))
        bg_color = input("Enter the background color for the canvases: ")
        return width, height, bg_color

    def normalize(self, coord, max_value, canvas_size):
        """
        Normalizes the coordinate value for drawing.
        """
        return int((coord + max_value / 2) * canvas_size / max_value)

    def draw_trajectory(self):
        """
        Draws the trajectory on three canvases (xy, xz, yz planes).
        """
        positions = self._positions
        root = tk.Tk()
        root.title("Trajectory Plot")

        width, height, bg_color = self.get_canvas_properties()

        canvas_xy = tk.Canvas(root, width=width, height=height, bg=bg_color)
        canvas_xy.grid(row=0, column=0)
        canvas_xz = tk.Canvas(root, width=width, height=height, bg=bg_color)
        canvas_xz.grid(row=0, column=1)
        canvas_yz = tk.Canvas(root, width=width, height=height, bg=bg_color)
        canvas_yz.grid(row=1, column=0)

        max_x = max(abs(positions[:, 0].min()), abs(positions[:, 0].max()))
        max_y = max(abs(positions[:, 1].min()), abs(positions[:, 1].max()))
        max_z = max(abs(positions[:, 2].min()), abs(positions[:, 2].max()))

        for pos in positions:
            x, y, z = pos

            canvas_xy.create_oval(self.normalize(x, max_x, width) - 2, self.normalize(y, max_y, height) - 2,
                                  self.normalize(x, max_x, width) + 2, self.normalize(y, max_y, height) + 2, fill="black")

            canvas_xz.create_oval(self.normalize(x, max_x, width) - 2, self.normalize(z, max_z, height) - 2,
                                  self.normalize(x, max_x, width) + 2, self.normalize(z, max_z, height) + 2, fill="red")

            canvas_yz.create_oval(self.normalize(y, max_y, width) - 2, self.normalize(z, max_z, height) - 2,
                                  self.normalize(y, max_y, width) + 2, self.normalize(z, max_z, height) + 2, fill="blue")

        root.mainloop()

