import random
import sys
import time
from typing import Tuple, List

import gym
import pyglet
from envs.custom_tol_env_dir.tol_2d.ball_coordinates_grid import state_ball_mapper
from envs.custom_tol_env_dir.tol_2d.mapping import int_to_state, state_to_int
from envs.custom_tol_env_dir.tol_2d.state import TolState
from envs.custom_tol_env_dir.tol_2d.tol_2d_view import rod1_line_coordinates, \
    rod2_line_coordinates, rod3_line_coordinates, horizontal_line_coordinates, active_ball_coordinates, \
    goal_ball_coordinates, WINDOW_WIDTH, WINDOW_HEIGHT, START_X, radius, tol_height, increment
from gym.envs.classic_control import rendering

"""
CONSTANTS
"""
MIN = 1
MAX = 6

MIN_MOVES = {
    (53, 14): 4,
    (11, 52): 5,
    (34, 56): 5,
    (46, 16): 6,
    (33, 52): 6,
    (13, 32): 6,
    (16, 25): 4,
    (24, 66): 5,
    (33, 54): 4,
    (23, 62): 6,
    (16, 46): 6,
    (51, 32): 5,
}


class ToLTaskEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, start_state=16, goal_state=25):
        """
        Action space: Represented as (colour permutation number, arrangement number)
        Colour Permutation Number: there are 6 different ways how 3 colour balls can be
        arranged in the same spatial arrangement
        Arrangement number : There are 6 possible ways how 3 balls can be arranged in the
        task spatially.
        """
        super(ToLTaskEnv, self).__init__()

        self.initial_state = start_state
        self.state = start_state
        self.goal_state = goal_state

        self.counter = 0
        self.is_done = False
        self.reward = 0
        self.info = None
        self.__version__ = "0.0.1"
        self.observation_space = gym.spaces.MultiDiscrete([(1, 6), (1, 6)])
        self.viewer = None
        self.delay = 1
        self.goal_positions = state_ball_mapper.get(int_to_state.get(self.goal_state))
        self.min_moves = MIN_MOVES.get((self.initial_state, self.goal_state))
        self.goal_red = self.goal_positions.red
        self.goal_green = self.goal_positions.green
        self.goal_blue = self.goal_positions.blue

        self.label_color = (10, 20, 255, 255)
        self.active_task_label = None
        self.goal_task_label = None

        self.red_ball = None
        self.green_ball = None
        self.blue_ball = None
        self.transform_red = None
        self.transform_green = None
        self.transform_blue = None
        self.no_goal_positions = self.no_in_positions(self.state)

        print('Env initialized')

    @property
    def ball_positions(self):
        return state_ball_mapper.get(int_to_state.get(self.state))

    @property
    def red_coordinates(self):
        return active_ball_coordinates.get_position_coordinates(self.red)

    @property
    def green_coordinates(self):
        return active_ball_coordinates.get_position_coordinates(self.green)

    @property
    def blue_coordinates(self):
        return active_ball_coordinates.get_position_coordinates(self.blue)

    @property
    def goal_red_coordinates(self):
        return goal_ball_coordinates.get_position_coordinates(self.goal_red)

    @property
    def goal_green_coordinates(self):
        return goal_ball_coordinates.get_position_coordinates(self.goal_green)

    @property
    def goal_blue_coordinates(self):
        return goal_ball_coordinates.get_position_coordinates(self.goal_blue)

    @property
    def red(self):
        return self.ball_positions.red

    @property
    def green(self):
        return self.ball_positions.green

    @property
    def blue(self):
        return self.ball_positions.blue

    @staticmethod
    def clamp(n) -> int:
        """
        Used in action space to adjust observation space values
        to always be between MIN and MAX which is between 1 and
        6 for this task
        :param n: number that may need to ba adjusted
        :return: Number between MIN and MAX included
        """
        if n < MIN:
            return MAX
        elif n > MAX:
            return MIN
        else:
            return n

    @property
    def action_space(self) -> List:
        """
        Action space is dependant from current state
        :return: List of tuples that represent actions that can be taken
        """
        color_permutation_no = int_to_state.get(self.state).permutation_no
        arrangement = int_to_state.get(self.state).arrangement_number
        possible_actions = {
            1: [(color_permutation_no * 10 + 2),
                (color_permutation_no * 10 + 3)],

            2: [(color_permutation_no * 10 + 1),
                (color_permutation_no * 10 + 3),

                state_to_int.get(TolState(
                    (self.clamp(color_permutation_no - 1),
                     self.clamp(color_permutation_no + 1)
                     )[color_permutation_no % 2 == 1], 5
                ))
                ],

            3: [(color_permutation_no * 10 + 1),
                (color_permutation_no * 10 + 2),
                (color_permutation_no * 10 + 4),
                (color_permutation_no * 10 + 5)],

            4: [(color_permutation_no * 10 + 3),
                (color_permutation_no * 10 + 5),
                state_to_int.get(TolState(

                    (self.clamp(color_permutation_no + 1), self.clamp(color_permutation_no - 1),
                     )[color_permutation_no % 2 == 1], 6
                ))
                ],

            5: [(color_permutation_no * 10 + 3),
                (color_permutation_no * 10 + 4),
                (color_permutation_no * 10 + 6),
                state_to_int.get(TolState(

                    (self.clamp(color_permutation_no - 1),
                     self.clamp(color_permutation_no + 1)
                     )[color_permutation_no % 2 == 1], 2
                ))
                ],

            6: [(color_permutation_no * 10 + 5),
                state_to_int.get(TolState(

                    (self.clamp(color_permutation_no + 1),
                     self.clamp(color_permutation_no - 1)
                     )[color_permutation_no % 2 == 1], 4
                ))
                ]

        }[arrangement]
        return possible_actions

    def __find_closest_goal_arrangement_action(self) -> int:
        """
        Helper function to find out the one action from the action space
        should be rewarded based in ball arrangement number. Finds action
        in the action space with ball arrangement colour number that is
        the closest to the goal state arrangement number.
        :return: Action from the action space
        """
        result = None
        candidates = [a for a in self.action_space if a // 10 == self.goal_state // 10]
        if len(candidates) == 1:
            result = candidates[0]
        else:
            candidates = [a % 10 for a in candidates]
            closest_arrangement = min(candidates, key=lambda x: abs(x - self.goal_state % 10))

            for action in candidates:
                if action % 10 == closest_arrangement:
                    result = action
        return result

    def __find_closest_goal_colour_action(self) -> int:
        """
        Helper function to find out which action from action space
        should be rewarded based on colour permutation number. The
        permutation number in action space that is closer to the
        goal permutation number is rewarded.
        :return: Action from the action space
        """
        action_space_colours = [a // 10 for a in self.action_space]
        closest_colour = min(action_space_colours, key=lambda x: abs(x - self.goal_state // 10))
        for action in self.action_space:
            if action // 10 == closest_colour:
                return action
            else:
                return 0

    def _get_rewarded_action(self) -> int:
        """
        Gets all actions from the action space and determines
        which action leads closer to the goal state and, therefore,
        should be rewarded.
        :return: Action from action_space that should be rewarded
        """
        is_goal_in_action_space = self.goal_state in self.action_space
        if is_goal_in_action_space:
            return self.goal_state

        goal_colour_no = self.goal_state // 10
        has_actions_with_goal_colour_no = goal_colour_no in [a // 10 for a in self.action_space]

        if has_actions_with_goal_colour_no:
            return self.__find_closest_goal_arrangement_action()
        return self.__find_closest_goal_colour_action()

    def _calculate_reward(self, action: int) -> int:
        """
        Only one action from all the possible actions that can be taken
        has a positive reward. All other actions have a negative reward.
        First, it calculates which action is leads closer to the the goal
        state and rewards that action.
        In order to calculate the action:
        1) Determine if the action leads directly to the goal state? If so,
        this action is rewarded.
        2) Else check if any action has the goal state colour permutation
        number if yes then finds which of these actions leads closer to the
        goal ball arrangement and selects that action.
        3) Else determines which action is the closest to goal states colour
        permutation number and selects it.
        :param action: action to use to calculate reward
        :return: +100 if action is rewarded, -100 is not
        """
        action_to_reward = self._get_rewarded_action()
        print(f'Action to reward: {action_to_reward}')
        print(f'Action is {action}')
        if action == action_to_reward:
            print('Reward: +100')
            return +100
        else:
            print('Reward: -100')
            return -100

    def get_reward(self, action):
        """
        Positive reward when any of the balls are put in the
        goal position. Negative reward when any of the balls
        that was before in goal position is moved away
        :param action:
        :return:
        """
        before = self.no_goal_positions
        balls_in_goal_place = self.no_in_positions(action)
        self.no_goal_positions = balls_in_goal_place
        print('Before', before)
        print('Balls in goal', balls_in_goal_place)
        moves_made = self.counter
        if action == self.goal_state:
            if self.counter == self.min_moves:
                return 100 / moves_made
            return 1 / moves_made
        if balls_in_goal_place > before:
            return 0.25 / moves_made
        elif balls_in_goal_place == before:
            return 0
        else:
            return (-1) * 0.25 / moves_made

    def no_in_positions(self, action):
        # get current ball positions
        a = int_to_state.get(action)
        red, green, blue = state_ball_mapper.get(a)

        # get goal ball positions
        goal = int_to_state.get(self.goal_state)
        red_goal, green_goal, blue_goal = state_ball_mapper.get(goal)

        # Init counter
        balls_in_goal_place = 0

        if red == red_goal:
            balls_in_goal_place += 1

        if green == green_goal:
            balls_in_goal_place += 1

        if blue == blue_goal:
            balls_in_goal_place += 1
        return balls_in_goal_place

    def get_random_action(self) -> int:
        """
        Randomly returns an action from action space
        :return: int
        """
        return random.choice(self.action_space)

    def is_game_complete(self):
        return self.state == self.goal_state

    def step(self, action: int) -> Tuple:
        """
        Runs one time-step of the environment's dynamics. The reset() method is called at the end of every episode
        :param action: The action to be executed in the environment
        :return: (observation, reward, done, info)
            observation (object):
                Observation from the environment at the current time-step
            reward (float):
                Reward from the environment due to the previous action performed
            done (bool):
                a boolean, indicating whether the episode has ended
            info (dict):
                a dictionary containing additional information about the previous action
        """
        if not self.is_done:
            self.counter += 1
            self.reward = self.get_reward(action)
            action_meaning = self.get_action_meaning(self.state, action)
            self.state = action
            time.sleep(self.delay)
            if self.is_game_complete():
                self.is_done = True

            self.info = {
                'initial_state': self.initial_state,
                'state': self.state,
                'goal_state': self.goal_state,
                'action_space': self.action_space,
                'red_position': self.red,
                'green_position': self.green,
                'blue_position': self.blue,
                'count': self.counter,
                'min_moves': self.min_moves,
                'action': action_meaning
            }
            print(f'Returning action={action}, reward={self.reward}, is_done={self.is_done}, info={self.info} ')
            print()
        return action, self.reward, self.is_done, self.info

    def reset(self):
        """
        Reset the environment state and returns an initial observation

        Returns
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        """
        self.state = self.initial_state
        self.counter = 0
        self.reward = 0
        self.info = None
        self.is_done = False
        return self.state

    def render(self, mode='human', close=False):

        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_WIDTH, WINDOW_HEIGHT)

            # Adding labels
            self._add_active_task_label(x=350, y=50)
            self._add_goal_task_label(x=300, y=350)
            self.active_task_label.text = f'Current state of the active task: {self.state}'
            self.goal_task_label.text = f'Goal Task state is: {self.goal_state}'

            """
            Active Task 
            """
            self._draw_task_frame(START_X)
            # Red
            self.red_ball = rendering.make_circle(radius)
            self.red_ball.set_color(250, 0, 0)
            self.transform_red = rendering.Transform(translation=self.red_coordinates)
            self.red_ball.add_attr(self.transform_red)
            self.viewer.add_geom(self.red_ball)

            # Green
            self.green_ball = rendering.make_circle(radius)
            self.green_ball.set_color(0, 99, 0)
            self.transform_green = rendering.Transform(translation=self.green_coordinates)
            self.green_ball.add_attr(self.transform_green)
            self.viewer.add_geom(self.green_ball)

            # Blue
            self.blue_ball = rendering.make_circle(radius)
            self.blue_ball.set_color(0, 0, 19)
            self.transform_blue = rendering.Transform(translation=self.blue_coordinates)
            self.blue_ball.add_attr(self.transform_blue)
            self.viewer.add_geom(self.blue_ball)

            """
            Goal Task 
            """
            self._draw_task_frame(START_X, tol_height + increment)
            self._add_goal_task()
        else:

            self.transform_red.set_translation(self.red_coordinates[0], self.red_coordinates[1])
            self.transform_green.set_translation(self.green_coordinates[0], self.green_coordinates[1])
            self.transform_blue.set_translation(self.blue_coordinates[0], self.blue_coordinates[1])
            self.active_task_label.text = f'Current state of the active task: {self.state}'

        return self.viewer.render()

    def _draw_task_frame(self, x, y_increment=0.0):
        l1 = rod1_line_coordinates(x, y_increment)
        l2 = rod2_line_coordinates(x, y_increment)
        l3 = rod3_line_coordinates(x, y_increment)
        hl = horizontal_line_coordinates(x, y_increment)
        line = rendering.Line(l1[0], l1[1])
        line2 = rendering.Line(l2[0], l2[1])
        line3 = rendering.Line(l3[0], l3[1])
        hor_line = rendering.Line(hl[0], hl[1])
        self.viewer.add_geom(line)
        self.viewer.add_geom(line2)
        self.viewer.add_geom(line3)
        self.viewer.add_geom(hor_line)

    def _add_goal_task(self):
        # Red
        red = rendering.make_circle(radius)
        red.set_color(250, 0, 0)
        transform = rendering.Transform(translation=self.goal_red_coordinates)
        red.add_attr(transform)
        self.viewer.add_geom(red)

        # Green
        green = rendering.make_circle(radius)
        green.set_color(0, 99, 0)
        transform = rendering.Transform(translation=self.goal_green_coordinates)
        green.add_attr(transform)
        self.viewer.add_geom(green)

        blue = rendering.make_circle(radius)
        blue.set_color(0, 0, 19)
        transform = rendering.Transform(translation=self.goal_blue_coordinates)
        blue.add_attr(transform)
        self.viewer.add_geom(blue)

    def _add_active_task_label(self, x, y):
        self.active_task_label = pyglet.text.Label(font_size=12,
                                                   x=x,
                                                   y=y,
                                                   anchor_x='center',
                                                   anchor_y='center',
                                                   color=self.label_color
                                                   )
        self.active_task_label.draw()
        self.viewer.add_geom(DrawText(self.active_task_label))

    def _add_goal_task_label(self, x, y):
        self.goal_task_label = pyglet.text.Label(font_size=12,
                                                 x=x,
                                                 y=y,
                                                 anchor_x='center',
                                                 anchor_y='center',
                                                 color=self.label_color
                                                 )
        self.goal_task_label.draw()
        self.viewer.add_geom(DrawText(self.goal_task_label))

    def get_action_meaning(self, s1, s2):
        s1 = int_to_state.get(s1)
        s2 = int_to_state.get(s2)
        r1, g1, b1 = state_ball_mapper.get(s1)
        r2, g2, b2 = state_ball_mapper.get(s2)
        red_moved = r2 - r1 != 0
        green_moved = g2 - g1 != 0
        blue_moved = b2 - b1 != 0
        colour = ''
        rod_from = ''
        rod_to = ''
        arr1 = None
        arr2 = None

        if red_moved:
            colour = 'R'
            arr1, arr2 = r1 % 10, r2 % 10
            rod_from = self.get_rod_letter(r1)
            rod_to = self.get_rod_letter(r2)

        if green_moved:
            colour = 'G'
            arr1, arr2 = g1 % 10, g2 % 10
            rod_from = self.get_rod_letter(g1)
            rod_to = self.get_rod_letter(g2)

        if blue_moved:
            colour = 'B'
            arr1, arr2 = b1 % 10, b2 % 10
            rod_from = self.get_rod_letter(b1)
            rod_to = self.get_rod_letter(b2)

        action = f'[{colour} {rod_from}{arr1} -> {colour} {rod_to}{arr2}]'
        return action

    def get_rod_letter(self, position):
        rod = position // 10
        if rod == 1:
            return 'L'
        if rod == 2:
            return 'C'
        if rod == 3:
            return 'R'
        return None

    def close(self):
        print(f'In close, {self.viewer}')
        if self.viewer and sys.meta_path:
            self.viewer.close()


class ToLTaskRewardsAfterCompletedEnv(ToLTaskEnv):

    def __init__(self, start_state=16, goal_state=25):
        super().__init__(start_state=start_state, goal_state=goal_state)

    def get_reward(self, action):
        if action == self.goal_state:
            if self.counter == self.min_moves:
                return 100
            return 1
        return 0


class DrawText:
    """
    Class to be able to add Labels with text on the canvas
    screen
    """

    def __init__(self, label: pyglet.text.Label):
        self.label = label

    def render(self):
        self.label.draw()
