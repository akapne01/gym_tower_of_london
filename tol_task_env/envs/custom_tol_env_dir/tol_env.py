import gym
import tkinter as tk

from envs.custom_tol_env_dir.tol_2d.tol_2d_view import TowerOfLondonTask


class ToLTaskEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, enable_render=True):
        super(ToLTaskEnv, self).__init__()

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

        self.viewer = None
        self.enable_render = enable_render

        self.state = []
        self.counter = 0
        self.done = 0
        self.reward = 0

        self.__version__ = "0.0.1"
        # Modify the observation space, low, high and shape values according to your custom environment's needs
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,))
        # Modify the action space, and dimension according to your custom environment's needs
        # self.action_space = gym.spaces.Box(4)
        print('Env initalized')

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
        return
