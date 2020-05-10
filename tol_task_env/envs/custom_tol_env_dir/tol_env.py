import gym
import tkinter as tk
from gym.spaces import Discrete

from envs.custom_tol_env_dir.tol_2d.tol_2d_view import TowerOfLondonTask


class ToLTaskEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, enable_render=True):
        """
        Action space: Represented as (colour permutation number, arrangement number)
        Colour Permutation Number: there are 6 differnt ways how 3 colour balls can be
        arranged in the same spatial arrangement
        Arrangement number : There are 6 possible ways how 3 balls can be arranged in the
        task spatially.
        :param enable_render:
        """
        super(ToLTaskEnv, self).__init__()
        self.root = None
        self.moves_counter = None
        self.result_task = None
        self.active_task = None

        self.viewer = None
        self.enable_render = enable_render
        self.state = (1, 6)
        self.counter = 0
        self.is_done = False
        self.reward = 0
        self.__version__ = "0.0.1"

        self.observation_space = gym.spaces.MultiDiscrete([(1, 6), (1, 6)])
        print('Env initialized')

    @staticmethod
    def clamp(n):
        minn = 1
        maxn = 6
        if n < minn:
            return maxn
        elif n > maxn:
            return minn
        else:
            return n

    @property
    def action_space(self):
        """
        Action space is dependant from current state
        :return: List of tuples that represent actions that can be taken
        """
        color_permutation_no = self.state[0]
        arrangement = self.state[1]
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

    def _set_up_task_window(self):
        root = tk.Tk()
        root.geometry('1000x700')
        root.title('Reinforcement Learning - Tower of London Task')
        self.moves_counter = tk.IntVar()
        self.result_task = TowerOfLondonTask(self.root, row_on_canvas=1)
        self.active_task = TowerOfLondonTask(self.root, row_on_canvas=0, end_task=self.result_task,
                                             moves_counter=self.moves_counter)
        self.moves_counter.set('Number of moves: {}'.format(self.active_task.frame.no_moves))
        label = tk.Label(self.root, textvariable=self.moves_counter)
        label.grid(row=0, column=3, sticky=tk.W)
        button_move = tk.Button(self.root, text='Next random move',
                                command=lambda: self.active_task.random_move(self.result_task, self.moves_counter))
        button_move.grid(row=0, column=2, sticky=tk.W)

        button_move_randomly = tk.Button(self.root, text='Keep moving randomly',
                                         command=(lambda: self.root.after(2000, self.active_task.move_randomly)))
        button_move_randomly.grid(row=0, column=1, sticky=tk.W)
        return root

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
        """

        :param mode:
        :return:
        """
        self.root = self._set_up_task_window()
        self.root.mainloop()
