import queue
import threading
import tkinter as tk
from typing import Tuple

import gym
from envs.custom_tol_env_dir.tol_2d.ball_coordinates_grid import ObservationSpaceCoordinates
from envs.custom_tol_env_dir.tol_2d.state import TolState
from envs.custom_tol_env_dir.tol_2d.tol_2d_view import TowerOfLondonTask, ball_coordinates, TolMainScreen

"""
CONSTANTS AND VARIABLES
"""
COLOUR_NO = 0
ARRANGEMENT_NO = 1
MIN = 1
MAX = 6


# WINDOW_WIDTH = 1000
# WINDOW_HEIGHT = 700
# TOL_RATIO = 2 / 3
# PADDING = 50
# START_X = 50
#
# tol_height = (WINDOW_HEIGHT - (PADDING * 4)) / 2
# tol_width = tol_height / TOL_RATIO
#
# radius = (tol_width - tol_height) / 5
# increment = (tol_width - tol_height) / 2
#
# """
# Holds all 16 possible ball position coordinates
# """
# ball_coordinates = ObservationSpaceCoordinates(START_X + increment, tol_height, TOL_RATIO)


class ToLTaskEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, enable_render=True):
        """
        Action space: Represented as (colour permutation number, arrangement number)
        Colour Permutation Number: there are 6 different ways how 3 colour balls can be
        arranged in the same spatial arrangement
        Arrangement number : There are 6 possible ways how 3 balls can be arranged in the
        task spatially.
        :param enable_render:
        """
        super(ToLTaskEnv, self).__init__()

        # self.root = tk.Tk()
        self.state = TolState(1, 6)
        self.end_state = TolState(2, 5)
        self.main = TolMainScreen(state=self.state, end_state=self.end_state)
        # self.active_task = TowerOfLondonTask(self.root, row_on_canvas=0, state=self.state,
        #                                      coordinates=ball_coordinates)
        # self.result_task = TowerOfLondonTask(self.root, row_on_canvas=1, state=self.end_state,
        #                                      coordinates=ball_coordinates)
        self.active_task = self.main.active_task
        self.result_task = self.main.result_task
        self.counter = 0
        self.is_done = False
        self.reward = 0
        self.__version__ = "0.0.1"
        self.observation_space = gym.spaces.MultiDiscrete([(1, 6), (1, 6)])
        self.enable_render = enable_render
        # if enable_render:
        #     self.render()

        print('Env initialized')

    @staticmethod
    def clamp(n):
        if n < MIN:
            return MAX
        elif n > MAX:
            return MIN
        else:
            return n

    @property
    def action_space(self):
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

    def __find_closest_goal_arrangement_action(self):
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

    def __find_closest_goal_colour_action(self):
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
        this action is rewarded
        2) Else finds action which has a closest colour permutation number
        and select that action.
        :param action: action to use to calculate reward
        :return: +100 if action is rewarded, -100 is not
        """
        action_to_reward = self._get_rewarded_action()
        if action == action_to_reward:
            print('Returning +100')
            return +100
        else:
            print('Returning -100')
            return -100

    def step(self, action):
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

    def render(self, mode='human', close=False):
        pass
