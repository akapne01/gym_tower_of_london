import random

from envs.custom_tol_env_dir.tol_2d.ball_colour import Ball
from envs.custom_tol_env_dir.tol_2d.rod import Rod

ROD_1_SIZE = 1
ROD_2_SIZE = 2
ROD_3_SIZE = 3


class TaskFrame:
    """
    Task Frame  represents the Tower of London Rods
    """

    coloursToWords = {0: 'EMPTY', 1: 'RED  ', 2: 'GREEN', 3: 'BLUE '}

    def __init__(self):
        self.ball_colour_sequence = [0, 0, 0]
        self.frame = [Rod(ROD_1_SIZE), Rod(ROD_2_SIZE), Rod(ROD_3_SIZE)]
        self.no_moves = 0
        self.last_rod = -1
        self.top_balls = [self.frame[0].top_ball_colour, self.frame[1].top_ball_colour, self.frame[2].top_ball_colour]

    def move_ball_from_to(self, rod_from, rod_to):
        colour = self.frame[rod_from].top_ball_colour
        self.frame[rod_from].remove_top_ball()
        self.frame[rod_to].put_ball_with_colour(colour)
        self.last_rod = rod_to
        self.no_moves += 1
        return colour

    def can_move_ball_to(self, rod):
        return not self.frame[rod].is_full

    def can_move_ball_from(self, rod):
        return self.frame[rod].next_free_index != 0

    def init_frame_with_random_balls(self):
        self.ball_colour_sequence = self.create_random_sequence_from_all_balls()
        for colour in self.ball_colour_sequence:
            is_ball_placed = True
            while is_ball_placed:
                rod_to_place_the_ball = random.choice(range(3))
                if self.can_move_ball_to(rod_to_place_the_ball):
                    self.frame[rod_to_place_the_ball].put_ball_with_colour(colour)
                    is_ball_placed = False

    def create_random_sequence_from_all_balls(self):
        """
        Creates a random sequence from all balls in Ball colour enum
        """
        random_sequence_balls = []
        balls_list = list(Ball)
        for i in range(len(balls_list)):
            ball_to_add = random.choice(balls_list)
            balls_list.remove(ball_to_add)
            random_sequence_balls.append(ball_to_add.value)
        return random_sequence_balls

    def print_number_grid(self):
        r_1 = self.frame[0].balls_on_rod
        r_2 = self.frame[1].balls_on_rod
        r_3 = self.frame[2].balls_on_rod
        print("       ", r_3[2])
        print("   ", r_2[1], " ", r_3[1])
        print(r_1[0], " ", r_2[0], " ", r_3[0])

    def print_word_grid(self):
        r_1 = self.frame[0].balls_on_rod
        r_2 = self.frame[1].balls_on_rod
        r_3 = self.frame[2].balls_on_rod
        print("               ", self.coloursToWords.get(r_3[2]))
        print("       ", self.coloursToWords.get(r_2[1]), " ", self.coloursToWords.get(r_3[1]))
        print(self.coloursToWords.get(r_1[0]), " ", self.coloursToWords.get(r_2[0]), " ",
              self.coloursToWords.get(r_3[0]))

    def is_equals_to(self, task_frame):
        for i, j in zip(task_frame.frame, self.frame):
            if i.balls_on_rod != j.balls_on_rod:
                print('False')
                return False
        print('True')
        return True

    def move_from(self):
        move_not_found = True
        move = -1
        while move_not_found:
            rod_from = [0, 1, 2]
            if self.last_rod == -1:
                move = random.choice(rod_from)
            else:
                if len(self.frame[2].balls_on_rod) != 3:
                    rod_from.remove(self.last_rod)
                move = random.choice(rod_from)
            if self.can_move_ball_from(move):
                move_not_found = False
        return move

    def move_to(self, move_from):
        not_found = True
        move_to = -1
        rod_to = [0, 1, 2]
        rod_to.remove(move_from)
        while not_found:
            move_to = random.choice(rod_to)
            print('Move to: ', move_to)
            if not self.frame[move_to].is_full:
                not_found = False
        return move_to

    def make_a_random_move(self):
        move_from = self.move_from()
        move_to = self.move_to(move_from)
        colour = self.move_ball_from_to(move_from, move_to)
        return move_from, move_to, colour


if __name__ == '__main__':
    g = TaskFrame()
    g.frame[0].put_ball_with_colour(1)
    g.frame[1].put_ball_with_colour(2)
    g.frame[2].put_ball_with_colour(3)
    g.print_number_grid()

    print('Move Randomly')
    # return from & to
    g.make_a_random_move()
    g.print_number_grid()

    g.make_a_random_move()
    print('After random move 1')
    g.print_number_grid()
    g.make_a_random_move()
    print('After random move 2')
    g.print_number_grid()
