import random
from typing import List, Tuple

import gym
import envs  # needed to run
import pydot
from anytree import Node
from anytree.exporter import DotExporter

from training.utils.planning_helper import get_possible_actions
from training.utils.tree_methods import calculate_step_reward

"""
Parameters
"""
max_reward = 100
alpha = 0.5
gamma = 0.5
lambda_ = 0.5

# Max possible reward to episode.
c_p = 110


class SarsaUCTIteration:
    visited_states = set()

    def __init__(self, environment):
        # self.tree = construct_reward_planning_tree(state=environment.state, goal_state=environment.goal_state,
        #                                            start_position=environment.initial_state,
        #                                            moves_made=environment.counter)
        self.env = environment

    def generate_episode(self, state: int) -> List:
        """
        Generates an episode that represents how the episode
        was played out by the agent.
        Episode consists of the list of tuples where each tuple
        represents information about one action taken from the
        state.
        Each tuple has next_state, reward and action meaning
        :param state: int that represents the state
        :return: List of tuples
        """
        terminal_state_not_reached = True
        episode = []
        s = state
        while terminal_state_not_reached:
            action = self.tree_policy(s, episode)
            next_state, reward, done, info = env.step(action)
            env.render()
            action_meaning = info.get('action')
            step_result = (next_state, reward, action_meaning)
            episode.append(step_result)
            s = next_state
            if done:
                terminal_state_not_reached = False
        return episode

    def expand_tree(self, episode):
        # to write
        pass

    def backup_TD_errors(self, episode):
        pass

    def sarsa_uct_iteration(self, state):
        episode = self.generate_episode(state)
        self.expand_tree(episode)
        self.backup_TD_errors(episode)

    def tree_policy(self, state: int, episode: List) -> int:
        """
        If state has been visited before then it is already added to the tree and action
        is selected using UCB1 tree policy.
        If action has not been visited before then action is selected randomly and state
        transitions are simulated from this state and appended to the episode.
        :param episode:
        :param state:
        :return:
        """
        if state in self.visited_states:
            return self.UCB1_tree_policy(state)
        else:
            """
            Might be able to add the for loop here to represent the 
            search depth.
            """
            action = random.choice(self.get_possible_actions(state))
            simulation = self.simulate_step_transition(state, action)
            episode.append(simulation)
        return action

    def UCB1_tree_policy(self, state: int) -> int:
        """

        :param state:
        :return:
        """
        # aim is to apply multi armed bandit problem for each action separately

        # loop through possible actions
        for action in self.get_possible_actions(state):

            # if state is visited; then normalize and calculate q
            if action in self.visited_states:
                # tee(s_i).V
                v = 1
                value_norm = self.normalize(v)
            else:
                pass
        return 0

    def normalize(self, value):
        """
        UCB1 selection policy requires values normalized in the
        interval [0, 1].
        :param value:
        :return:
        """
        print('Returning', value / c_p)
        return value / c_p

    def get_possible_actions(self, state: int) -> List:
        """
        Takes a number denoting the state and returns
        back List with all the possible actions that
        can be taken from this state.
        """
        return get_possible_actions(state)

    def simulate_step_transition(self, state: int, action: int) -> Tuple:
        """
        This function simulates step function from environment.
        Corresponds to panning ahead in the mind and executing
        these actions in imagination instead of the real
        environment.
        :param state: int number representing the state
        :param action: int number representing the action,
        this number corresponds to the next state.
        :return:
        """
        goal_state = self.env.goal_state
        moves_made = self.env.counter
        reward = self.get_reward(action, moves_made=moves_made, state=state)
        is_done = action == goal_state
        action_meaning = self.env.get_action_meaning(state, action)
        info = {
            'state': state,
            'action': action_meaning
        }
        print(
            f'Returning action={action}, reward={reward}, is_done={is_done}, info={info} ')
        print()

        return action, reward, is_done, info

    def get_reward(self, action: int, moves_made: int, state: int) -> float:
        return calculate_step_reward(action=action,
                                     goal_state=self.env.goal_state,
                                     start_position=self.env.initial_state,
                                     moves_made=moves_made, state=state)


def save_tree(tree: Node) -> None:
    """
    Saves tree as a *.dot file then reads the
    dot file that is then converted to a png
    file
    :param tree: Node to save
    """
    DotExporter(tree,
                nodenamefunc=lambda node: node.name,
                edgeattrfunc=lambda parent,
                                    child: "style=bold,label=%10.2f" % (
                            child.reward or 0)
                ).to_dotfile(f"images/dot/reward_tree_{s}.dot")
    (graph,) = pydot.graph_from_dot_file(f"images/dot/reward_tree_{s}.dot")
    graph.write_png(f"images/png/reward_tree_{s}.png")


if __name__ == '__main__':
    s = 16
    env = gym.make('TolTask-v0', start_state=s, goal_state=25)
    rl = SarsaUCTIteration(env)
    state = env.reset()
    # env.render()
    # rl.generate_episode(env=env, state=state)
