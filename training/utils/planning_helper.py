import random
from typing import List

import numpy as np
from networkx.tests.test_convert_pandas import pd

from envs.custom_tol_env_dir import ToLTaskEnv
from envs.custom_tol_env_dir.tol_2d.mapping import int_to_state, state_to_int
from envs.custom_tol_env_dir.tol_2d.state import TolState

# from training.utils.tree_methods import plan_for_best_actions

FILL_VALUE = -100


def init_q_table():
    poss_states = 36
    poss_actions = 36
    q_table = pd.DataFrame(
        np.array([[FILL_VALUE] * poss_actions] * poss_states))
    int_states = int_to_state.keys()  # contains all possible states
    q_table.index = int_states
    q_table.columns = int_states
    for i in int_states:
        actions = get_possible_actions(i)
        for a in actions:
            q_table.loc[a, i] = 0
    return q_table


def get_best_Q_value(state, Q) -> float:
    """
    Gets the next actions and find the
    max of the action Q values for specified
    state. Returns max Q-value
    :param state:
    :param Q:
    :return:
    """
    max_q = 0
    max_a = 0
    best = []
    # all possible actions
    actions = get_possible_actions(state)
    df = Q.loc[actions]
    max_s = df[state].max()  # should return  max_s?
    # print(f'# get_best_Q :  state={state} ; max={max_s}')
    # for a in actions:
    #     value = Q.loc[a, state]
    #     if value == max_s:
    #         best.append(a)
    print(f'# get_best_Q: the best Q-value for s={state} is next_q={max_s}')
    return max_s


def get_possible_actions(state):
    color_permutation_no = int_to_state.get(state).permutation_no
    arrangement = int_to_state.get(state).arrangement_number
    possible_actions = {
        1: [(color_permutation_no * 10 + 2),
            (color_permutation_no * 10 + 3)],

        2: [(color_permutation_no * 10 + 1),
            (color_permutation_no * 10 + 3),

            state_to_int.get(TolState(
                (ToLTaskEnv.clamp(color_permutation_no - 1),
                 ToLTaskEnv.clamp(color_permutation_no + 1)
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

                (ToLTaskEnv.clamp(color_permutation_no + 1),
                 ToLTaskEnv.clamp(color_permutation_no - 1),
                 )[color_permutation_no % 2 == 1], 6
            ))
            ],

        5: [(color_permutation_no * 10 + 3),
            (color_permutation_no * 10 + 4),
            (color_permutation_no * 10 + 6),
            state_to_int.get(TolState(

                (ToLTaskEnv.clamp(color_permutation_no - 1),
                 ToLTaskEnv.clamp(color_permutation_no + 1)
                 )[color_permutation_no % 2 == 1], 2
            ))
            ],

        6: [(color_permutation_no * 10 + 5),
            state_to_int.get(TolState(

                (ToLTaskEnv.clamp(color_permutation_no + 1),
                 ToLTaskEnv.clamp(color_permutation_no - 1)
                 )[color_permutation_no % 2 == 1], 4
            ))
            ]

    }[arrangement]
    return possible_actions


def _find_closest_goal_arrangement_action(goal_state, possible_actions) -> int:
    """
    Helper function to find out the one action from the action space
    should be rewarded based in ball arrangement number. Finds action
    in the action space with ball arrangement colour number that is
    the closest to the goal state arrangement number.
    :return: Action from the action space
    """
    result = None
    candidates = [a for a in possible_actions if a // 10 == goal_state // 10]
    if len(candidates) == 1:
        result = candidates[0]
    else:
        candidates = [a % 10 for a in candidates]
        closest_arrangement = min(candidates,
                                  key=lambda x: abs(x - goal_state % 10))

        for action in candidates:
            if action % 10 == closest_arrangement:
                result = action
    return result


def _find_closest_goal_colour_action(goal_state, possible_actions) -> int:
    """
    Helper function to find out which action from action space
    should be rewarded based on colour permutation number. The
    permutation number in action space that is closer to the
    goal permutation number is rewarded.
    :return: Action from the action space
    """
    action_space_colours = [a // 10 for a in possible_actions]
    closest_colour = min(action_space_colours,
                         key=lambda x: abs(x - goal_state // 10))
    for action in possible_actions:
        if action // 10 == closest_colour:
            return action
        else:
            return 0


def _get_rewarded_action(goal_state, possible_actions) -> int:
    """
    Gets all actions from the action space and determines
    which action leads closer to the goal state and, therefore,
    should be rewarded.
    :return: Action from action_space that should be rewarded
    """

    is_goal_in_action_space = goal_state in possible_actions
    if is_goal_in_action_space:
        return goal_state

    goal_colour_no = goal_state // 10
    has_actions_with_goal_colour_no = goal_colour_no in [a // 10 for a in
                                                         possible_actions]

    if has_actions_with_goal_colour_no:
        return _find_closest_goal_arrangement_action(goal_state,
                                                     possible_actions)
    return _find_closest_goal_colour_action(goal_state, possible_actions)


def calculate_reward(state, action, goal_state: int) -> int:
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
    :param goal_state:
    :param state:
    :param action: action to use to calculate reward
    :return: +100 if action is rewarded, -100 is not
    """
    possible_actions = get_possible_actions(state)
    action_to_reward = _get_rewarded_action(goal_state, possible_actions)
    print(f'Action to reward: {action_to_reward}')
    print(f'Action is {action}')
    print(f'State is {state}')
    if action == action_to_reward:
        print('Reward: +100')
        return +100
    else:
        print('Reward: -100')
        return -100


state_ball_mapper = {
    11: (31, 32, 33),
    12: (31, 32, 11),
    13: (31, 32, 21),
    14: (31, 22, 21),
    15: (31, 11, 21),
    16: (22, 11, 21),
    21: (31, 33, 32),
    22: (31, 11, 32),
    23: (31, 21, 32),
    24: (31, 21, 22),
    25: (31, 21, 11),
    26: (22, 21, 11),
    31: (32, 32, 31),
    32: (32, 11, 31),
    33: (32, 21, 31),
    34: (22, 21, 31),
    35: (11, 21, 31),
    36: (11, 21, 22),
    41: (33, 32, 31),
    42: (11, 32, 31),
    43: (21, 32, 31),
    44: (21, 22, 31),
    45: (21, 11, 31),
    46: (21, 11, 22),
    51: (33, 31, 32),
    52: (11, 31, 32),
    53: (21, 31, 32),
    54: (21, 31, 22),
    55: (21, 31, 11),
    56: (21, 22, 11),
    61: (32, 31, 33),
    62: (32, 31, 11),
    63: (32, 31, 21),
    64: (22, 31, 21),
    65: (11, 31, 21),
    66: (11, 22, 21),
}
