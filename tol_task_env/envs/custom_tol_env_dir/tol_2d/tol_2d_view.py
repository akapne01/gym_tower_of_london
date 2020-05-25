import tkinter as tk
from typing import Tuple

from envs.custom_tol_env_dir.tol_2d.ball_coordinates_grid import ObservationSpaceCoordinates, state_ball_mapper
from envs.custom_tol_env_dir.tol_2d.state import XYPositions

"""
CONSTANTS AND VARIABLES
"""
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
TOL_RATIO = 2 / 3
PADDING = 50
START_X = 50

tol_height = (WINDOW_HEIGHT - (PADDING * 4)) / 2
tol_width = tol_height / TOL_RATIO

radius = (tol_width - tol_height) / 5
increment = (tol_width - tol_height) / 2

"""
Holds all 16 possible ball position coordinates 
"""
ball_coordinates = ObservationSpaceCoordinates(START_X + increment, tol_height, TOL_RATIO)

"""
FUNCTIONS TO DRAW TASK LINES
"""


def rod1_line_coordinates(x) -> Tuple:
    start = XYPositions(x=(x + increment), y=(x + tol_height - increment * 3))
    end = XYPositions(x=x + increment, y=x + tol_height - increment * 2)
    return start, end


def rod2_line_coordinates(x):
    start = XYPositions(x=x + increment * 3, y=x + tol_height - increment * 3)
    end = XYPositions(x=x + increment * 3, y=x + tol_height - increment)
    return start, end


def rod3_line_coordinates(x):
    start = XYPositions(x=x + increment * 5, y=x + tol_height - increment * 3)
    end = XYPositions(x=x + increment * 5, y=x + tol_height)
    return start, end


def horizontal_line_coordinates(x):
    start = XYPositions(x=x, y=x + increment)
    end = XYPositions(x=x + tol_width, y=x + increment)
    return start, end


class TowerOfLondonTask:
    """
    Creates Tower of London task screen.
    """

    def __init__(self, root, row_on_canvas, state, coordinates):
        """
        Initialize Tower of London Task
        :param root: tkinter window
        :param row_on_canvas: placement on tkinter window
        :param state: named tuple TolState
        :param coordinates: object containing all coordinates of all
        16 observation space coordinates
        """
        self.task_canvas = tk.Canvas(root, width=tol_width + PADDING * 2, height=tol_height + PADDING * 2, bg='white')
        self.task_canvas.grid(row=row_on_canvas, column=0, sticky=tk.W)
        self.coordinates = coordinates
        self.state = state

        self.ball_positions = state_ball_mapper.get(state)

        self.red = self.ball_positions.red
        self.green = self.ball_positions.green
        self.blue = self.ball_positions.blue

        self.red_coordinates = self.coordinates.get_position_coordinates(self.red)
        self.green_coordinates = self.coordinates.get_position_coordinates(self.green)
        self.blue_coordinates = self.coordinates.get_position_coordinates(self.blue)

        self.red_ball_object = None
        self.green_ball_object = None
        self.blue_ball_object = None
        self.draw_balls()

    def draw_balls(self):
        self.red_ball_object = self.add_ball(self.red_coordinates, 'red')
        self.green_ball_object = self.add_ball(self.green_coordinates, 'green')
        self.blue_ball_object = self.add_ball(self.blue_coordinates, 'blue')

    def add_ball(self, coordinate_array, colour):
        x = coordinate_array.x
        y = coordinate_array.y
        r = coordinate_array.r
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        return self.task_canvas.create_oval(x0, y0, x1, y1, width=2, fill=colour)

    def remove_ball(self, ball):
        self.task_canvas.delete(ball)

    def clear_balls(self):
        self.task_canvas.delete(self.red_ball_object)
        self.task_canvas.delete(self.green_ball_object)
        self.task_canvas.delete(self.blue_ball_object)


class TolMainScreen:
    def __init__(self, state, end_state):
        self.root = tk.Tk()
        self.state = state
        self.end_state = end_state
        self.active_task = TowerOfLondonTask(self.root, row_on_canvas=0, state=self.state,
                                             coordinates=ball_coordinates)
        self.result_task = TowerOfLondonTask(self.root, row_on_canvas=1, state=self.end_state,
                                             coordinates=ball_coordinates)
        self._set_up_canvas()

    def _add_task_title(self):
        text = tk.StringVar()
        text.set('Active Task')
        goal_text = tk.StringVar()
        goal_text.set('Goal Task')
        title = tk.Label(self.root, textvariable=text)
        goal_title = tk.Label(self.root, textvariable=goal_text)
        title.grid(row=0, column=0, sticky=tk.NW)
        goal_title.grid(row=1, column=0, sticky=tk.NW)

    def _set_up_canvas(self):
        self.root.geometry('490x700')
        self.root.title('Reinforcement Learning - Tower of London Task')
        self._add_task_title()
