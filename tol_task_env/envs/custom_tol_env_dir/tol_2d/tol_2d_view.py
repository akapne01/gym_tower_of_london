from typing import Tuple

from envs.custom_tol_env_dir.tol_2d.ball_coordinates_grid import ObservationSpaceCoordinates
from envs.custom_tol_env_dir.tol_2d.state import XYPositions

"""
CONSTANTS AND VARIABLES
"""
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 700
TOL_RATIO = 2 / 3
PADDING = 50
START_X = 50

tol_height = (WINDOW_HEIGHT - (PADDING * 4)) / 2
tol_width = tol_height / TOL_RATIO

radius = (tol_width - tol_height) / 5
increment = (tol_width - tol_height) / 2

"""
Holds all 16 possible ball position coordinates or active and goal tasks
"""
active_x = START_X + increment
active_y = active_x + radius
goal_y = active_x + tol_height + increment + radius
active_ball_coordinates = ObservationSpaceCoordinates(x=active_x, y=active_y, height=tol_height, aspect_ratio=TOL_RATIO)
goal_ball_coordinates = ObservationSpaceCoordinates(x=active_x, y=goal_y, height=tol_height, aspect_ratio=TOL_RATIO)

"""
FUNCTIONS TO DRAW TASK LINES
"""


def rod1_line_coordinates(x, y_increment=0.0) -> Tuple:
    start = XYPositions(x=(x + increment), y=(x + tol_height - increment * 3) + y_increment)
    end = XYPositions(x=x + increment, y=x + tol_height - increment * 2 + y_increment)
    return start, end


def rod2_line_coordinates(x, y_increment=0.0):
    start = XYPositions(x=x + increment * 3, y=x + tol_height - increment * 3 + y_increment)
    end = XYPositions(x=x + increment * 3, y=x + tol_height - increment + +  y_increment)
    return start, end


def rod3_line_coordinates(x, y_increment=0.0):
    start = XYPositions(x=x + increment * 5, y=x + tol_height - increment * 3 + y_increment)
    end = XYPositions(x=x + increment * 5, y=x + tol_height + y_increment)
    return start, end


def horizontal_line_coordinates(x, y_increment=0.0):
    start = XYPositions(x=x, y=x + increment + y_increment)
    end = XYPositions(x=x + tol_width, y=x + increment + y_increment)
    return start, end
