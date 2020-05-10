from envs.custom_tol_env_dir.tol_2d.exceptions import RodAlreadyFullException, RodEmptyException


class Rod:
    def __init__(self, size):
        self.is_full = False
        self.balls_on_rod = size * [0]
        self.top_ball_colour = 0
        self.next_free_index = 0
        self.size = size

    def print_rod(self):
        print(self.balls_on_rod)

    def put_ball_with_colour(self, colour):
        if self.next_free_index == -1:
            raise RodAlreadyFullException
        if self.next_free_index != -1:
            self.balls_on_rod[self.next_free_index] = colour
            self.top_ball_colour = colour
            self.next_free_index += 1
            if self.size == self.next_free_index:
                self.next_free_index = -1
                self.is_full = True

    def remove_top_ball(self):
        if self.next_free_index == 0:
            raise RodEmptyException
        if self.is_full:
            self.is_full = False
        if self.next_free_index == -1:
            self.next_free_index = self.size - 1
        else:
            self.next_free_index -= 1
        self.balls_on_rod[self.next_free_index] = 0
        if self.next_free_index == 0:
            self.top_ball_colour = 0
        else:
            self.top_ball_colour = self.balls_on_rod[self.next_free_index - 1]
