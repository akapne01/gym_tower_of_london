import random
from typing import Tuple, List

import gym
import pyglet
from envs.custom_tol_env_dir.tol_2d.ball_coordinates_grid import ObservationSpaceCoordinates
from envs.custom_tol_env_dir.tol_2d.state import TolState
from envs.custom_tol_env_dir.tol_2d.tol_2d_view import rod1_line_coordinates, \
    rod2_line_coordinates, rod3_line_coordinates, horizontal_line_coordinates

"""
CONSTANTS AND VARIABLES
"""
COLOUR_NO = 0
ARRANGEMENT_NO = 1
MIN = 1
MAX = 6

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


class ToLTaskEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self):
        """
        Action space: Represented as (colour permutation number, arrangement number)
        Colour Permutation Number: there are 6 different ways how 3 colour balls can be
        arranged in the same spatial arrangement
        Arrangement number : There are 6 possible ways how 3 balls can be arranged in the
        task spatially.
        :param enable_render:
        """
        super(ToLTaskEnv, self).__init__()

        self.state = TolState(1, 6)
        self.goal_state = TolState(2, 5)
        self.counter = 0
        self.is_done = False
        self.reward = 0
        self.__version__ = "0.0.1"
        self.observation_space = gym.spaces.MultiDiscrete([(1, 6), (1, 6)])
        self.viewer = None
        print('Env initialized')

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
        color_permutation_no = self.state.permutation_no
        arrangement = self.state.arrangement_number
        possible_actions = {
            1: [(color_permutation_no, 2), (color_permutation_no, 3)],
            2: [(color_permutation_no, 1), (color_permutation_no, 3),
                ((self.clamp(color_permutation_no - 1), 5), (self.clamp(color_permutation_no + 1), 5))[
                    color_permutation_no % 2 == 1]],
            3: [(color_permutation_no, 1), (color_permutation_no, 2), (color_permutation_no, 4),
                (color_permutation_no, 5)],
            4: [(color_permutation_no, 3), (color_permutation_no, 5),
                ((self.clamp(color_permutation_no + 1), 6), (self.clamp(color_permutation_no - 1), 6))[
                    color_permutation_no % 2 == 1]],
            5: [(color_permutation_no, 3), (color_permutation_no, 4), (color_permutation_no, 6),
                ((self.clamp(color_permutation_no - 1), 2), (self.clamp(color_permutation_no + 1), 2))[
                    color_permutation_no % 2 == 1]],
            6: [(color_permutation_no, 5),
                ((self.clamp(color_permutation_no + 1), 4), (self.clamp(color_permutation_no - 1), 4))[
                    color_permutation_no % 2 == 1]]
        }[arrangement]
        return possible_actions

    def __find_closest_goal_arrangement_action(self) -> Tuple:
        """
        Helper function to find out the one action from the action space
        should be rewarded based in ball arrangement number. Finds action
        in the action space with ball arrangement colour number that is
        the closest to the goal state arrangement number.
        :return: Action from the action space
        """
        result = None
        candidates = [a for a in self.action_space if a[COLOUR_NO] == self.goal_state[COLOUR_NO]]
        if len(candidates) == 1:
            result = candidates[0]
        else:
            closest_arrangement = min(candidates[ARRANGEMENT_NO],
                                      key=lambda x: abs(x - self.goal_state[ARRANGEMENT_NO]))
            for action in candidates:
                if action[ARRANGEMENT_NO] == closest_arrangement:
                    result = action
        return result

    def __find_closest_goal_colour_action(self) -> Tuple:
        """
        Helper function to find out which action from action space
        should be rewarded based on colour permutation number. The
        permutation number in action space that is closer to the
        goal permutation number is rewarded.
        :return: Action from the action space
        """
        closest_colour = min(self.action_space[COLOUR_NO], key=lambda x: abs(x - self.goal_state[COLOUR_NO]))
        for action in self.action_space:
            if action[COLOUR_NO] == closest_colour:
                return action
            else:
                return None

    def _get_rewarded_action(self) -> Tuple:
        """
        Gets all actions from the action space and determines
        which action leads closer to the goal state and, therefore,
        should be rewarded.
        :return: Action from action_space that should be rewarded
        """
        is_goal_in_action_space = self.goal_state in self.action_space
        if is_goal_in_action_space:
            return self.goal_state

        goal_colour_no = self.goal_state[COLOUR_NO]
        has_actions_with_goal_colour_no = goal_colour_no in [a[COLOUR_NO] for a in self.action_space]

        if has_actions_with_goal_colour_no:
            return self.__find_closest_goal_arrangement_action()
        return self.__find_closest_goal_colour_action()

    def _calculate_reward(self, action: Tuple) -> int:
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
        if action == action_to_reward:
            print('Reward: +100')
            return +100
        else:
            print('Reward: -100')
            return -100

    def step(self, action: Tuple) -> Tuple:
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

        print('Taking step')
        # Implement your step method here
        #   - Calculate reward based on the action
        #   - Calculate next observation
        #   - Set done to True if end of episode else set done to False
        #   - Optionally, set values to the info dict
        # return (observation, reward, done, info)
        observation = None
        action = random.choice(self.action_space)
        reward = self._calculate_reward(action)
        info = None
        return observation, reward, self.is_done, info

    def reset(self):
        """
        Reset the environment state and returns an initial observation

        Returns
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        """
        # Implement your reset method here
        # return observation
        observation = None
        return observation

    def render(self, mode='human', close=False):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_WIDTH, WINDOW_HEIGHT)

            # Adding label

            self.acvtiva_task_label = pyglet.text.Label('Active Task', font_size=12,
                                                        x=400,
                                                        y=50,
                                                        # x=50, y=WINDOW_HEIGHT * 2.5 / 40.00,
                                                        anchor_x='center',
                                                        anchor_y='center',
                                                        color=(10, 20, 255, 255)
                                                        )
            self.acvtiva_task_label.text = f'Current state of the active task: {self.state}'
            self.acvtiva_task_label.draw()
            self.viewer.add_geom(DrawText(self.acvtiva_task_label))

            l1 = rod1_line_coordinates(START_X)
            l2 = rod2_line_coordinates(START_X)
            l3 = rod3_line_coordinates(START_X)
            hl = horizontal_line_coordinates(START_X)

            line = rendering.Line(l1[0], l1[1])
            line2 = rendering.Line(l2[0], l2[1])
            line3 = rendering.Line(l3[0], l3[1])
            hor_line = rendering.Line(hl[0], hl[1])
            self.viewer.add_geom(line)
            self.viewer.add_geom(line2)
            self.viewer.add_geom(line3)
            self.viewer.add_geom(hor_line)

            # Red
            red = rendering.make_circle(25)
            red.set_color(250, 0, 0)
            transform = rendering.Transform(translation=(237.5, 225.0))
            red.add_attr(transform)
            self.viewer.add_geom(red)

            # Green
            green = rendering.make_circle(25)
            green.set_color(0, 99, 0)
            transform = rendering.Transform(translation=(112.5, 275.0))
            green.add_attr(transform)
            self.viewer.add_geom(green)

            blue = rendering.make_circle(25)
            blue.set_color(0, 0, 19)
            transform = rendering.Transform(translation=(237.5, 275.0))
            blue.add_attr(transform)
            self.viewer.add_geom(blue)

        # return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        return self.viewer.render()


class DrawText:
    def __init__(self, label: pyglet.text.Label):
        self.label = label

    def render(self):
        self.label.draw()
