import tkinter as tk

from envs.custom_tol_env_dir.tol_2d.ball_coordinates_grid import ObservationSpaceCoordinates, state_ball_mapper
from envs.custom_tol_env_dir.tol_2d.state import TolState

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


def rod1_line_coordinates(x):
    return [x + increment, x + tol_height - increment, x + increment, x + tol_height]


def rod2_line_coordinates(x):
    return [x + increment * 3, x + tol_height - increment * 2, x + increment * 3,
            x + tol_height]


def rod3_line_coordinates(x):
    return [x + increment * 5, x + tol_height - increment * 3, x + increment * 5,
            x + tol_height]


def draw_line(xy_array, canvas):
    x1 = xy_array[0]
    y1 = xy_array[1]
    x2 = xy_array[2]
    y2 = xy_array[3]
    canvas.create_line(x1, y1, x2, y2)


def draw_task_lines(x, canvas):
    rod1 = rod1_line_coordinates(x)
    rod2 = rod2_line_coordinates(x)
    rod3 = rod3_line_coordinates(x)
    canvas.create_line(x, x + tol_height, x + tol_width, x + tol_height)  # horizontal
    draw_line(rod1, canvas)
    draw_line(rod2, canvas)
    draw_line(rod3, canvas)


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
        draw_task_lines(START_X, self.task_canvas)
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
        # self.root.mainloop()

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
