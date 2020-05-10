import tkinter as tk

from envs.custom_tol_env_dir.tol_2d.ball_colour import Ball
from envs.custom_tol_env_dir.tol_2d.ball_coordinates_grid import AllBallCoordinates
from envs.custom_tol_env_dir.tol_2d.task_frame import TaskFrame

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

ball_coordinates = AllBallCoordinates(START_X + increment, tol_height, TOL_RATIO).coordinate_matrix

"""
FUNCTIONS
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

    def __init__(self, root, row_on_canvas, balls=None, end_task=None, moves_counter=None):
        """
        Initialize Tower of London task screen
        :param root: Tinker window to use to display task
        :param row_on_canvas: 2 rows, one represents the result task, other the active task
        :param balls: grid representing current ball positions. If not provided, then selected randomly
        :param end_task: If provided then will be used to determine end stat
        :param moves_counter: tinker variable that is displayed as a label and represents number of moves
        taken
        """
        self.task_canvas = tk.Canvas(root, width=tol_width + PADDING * 2, height=tol_height + PADDING * 2, bg='white')
        self.task_canvas.grid(row=row_on_canvas, column=0, sticky=tk.W)
        draw_task_lines(START_X, self.task_canvas)
        self.frame = TaskFrame()
        if balls is None:
            self.frame.init_frame_with_random_balls()
        else:
            self.frame.frame = balls
        self.red_ball = 0
        self.green_ball = 0
        self.blue_ball = 0
        self.print_all_balls()
        self.is_result_found = False
        self.result_task = end_task
        self.var = moves_counter

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

    def print_all_balls(self):
        red_coordinates = self.get_red_ball_coordinates()
        self.red_ball = self.add_ball(ball_coordinates.get('{}{}'.format(red_coordinates[0], red_coordinates[1])),
                                      'red')
        green_coordinates = self.get_green_ball_coordinates()
        self.green_ball = self.add_ball(ball_coordinates.get('{}{}'.format(green_coordinates[0], green_coordinates[1])),
                                        'green')
        blue_coordinates = self.get_blue_ball_coordinates()
        self.blue_ball = self.add_ball(ball_coordinates.get('{}{}'.format(blue_coordinates[0], blue_coordinates[1])),
                                       'blue')

    def clear_balls(self):
        self.task_canvas.delete(self.red_ball)
        self.task_canvas.delete(self.green_ball)
        self.task_canvas.delete(self.blue_ball)

    def get_ball_coordinates(self, colour):
        grid = self.frame.frame
        counter_rod = 0
        for rod in grid:
            balls = rod.balls_on_rod
            counter_index = 0
            for r in balls:
                if r == colour:
                    return counter_rod, counter_index
                counter_index += 1
            counter_rod += 1

    def get_red_ball_coordinates(self):
        return self.get_ball_coordinates(Ball.RED.value)

    def get_blue_ball_coordinates(self):
        return self.get_ball_coordinates(Ball.BLUE.value)

    def get_green_ball_coordinates(self):
        return self.get_ball_coordinates(Ball.GREEN.value)

    def get_diff_x_and_y(self, rod_from, rod_to):
        x_diff = tol_height / TOL_RATIO - tol_height
        y_diff = radius * 2
        # from_index = self.frame.frame[rod_from].next_free_index
        r_from = self.frame.frame[rod_from].balls_on_rod
        r_to = self.frame.frame[rod_to].balls_on_rod
        from_index = 0
        for i in r_from:
            if i == 0:
                break
            from_index += 1

        to_index = -1
        for i in r_to:
            if i == 0:
                break
            to_index += 1

        # if from_index == -1:
        #     from_index = self.frame[rod_from].size
        # else:
        #     from_index -= 1
        # to_index = self.frame.frame[rod_to].next_free_index

        x = (rod_to - rod_from) * x_diff
        y = (from_index - to_index) * y_diff
        print('rod from', rod_from)
        print('rod_to', rod_to)
        print('from_index', from_index)
        print('to_index', to_index)
        print('x', x)
        print('y', y)
        return x, y

    def random_move(self, result_task, var):
        if not self.is_result_found:
            rod_from, rod_to, colour = self.frame.make_a_random_move()
            if self.frame.is_equals_to(result_task.frame):
                self.is_result_found = True
            # self.clear_balls()
            # self.print_all_balls()

            # assign ball object based on colour
            ball = 0
            if colour == 1:
                ball = self.red_ball
            elif colour == 2:
                ball = self.green_ball
            else:
                ball = self.blue_ball
            # Calculate x and y
            print('# colour', colour)
            print('# rod_from', rod_from)
            print('# rod_to', rod_to)
            x, y = self.get_diff_x_and_y(rod_from, rod_to)
            self.task_canvas.move(ball, x, y)
            var.set('Number of moves: {}'.format(self.frame.no_moves))

    def move_randomly(self):
        self.random_move(self.result_task, self.var)
        self.task_canvas.after(600, self.move_randomly)


class TolMainScreen:
    def __init__(self):
        root = tk.Tk()
        root.geometry('1000x700')
        root.title('Reinforcement Learning - Tower of London Task')
        moves_counter = tk.IntVar()
        result_task = TowerOfLondonTask(root, row_on_canvas=1)
        self.active_task = TowerOfLondonTask(root, row_on_canvas=0, end_task=result_task, moves_counter=moves_counter)
        moves_counter.set('Number of moves: {}'.format(self.active_task.frame.no_moves))
        label = tk.Label(root, textvariable=moves_counter)
        label.grid(row=0, column=3, sticky=tk.W)
        button_move = tk.Button(root, text='Next random move',
                                command=lambda: self.active_task.random_move(result_task, moves_counter))
        button_move.grid(row=0, column=2, sticky=tk.W)

        button_move_randomly = tk.Button(root, text='Keep moving randomly',
                                         command=(lambda: root.after(2000, self.active_task.move_randomly)))
        button_move_randomly.grid(row=0, column=1, sticky=tk.W)
        root.mainloop()

    def move_randomly(self):
        self.active_task.move_randomly()


if __name__ == '__main__':
    TolMainScreen()
