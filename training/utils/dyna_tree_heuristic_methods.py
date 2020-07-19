import random
from typing import List, Dict, Set

import pandas as pd

from training.utils.dyna_helper_methods import q_value_update, update_model
from training.utils.planning_helper import get_possible_actions
from training.utils.tree_methods import calculate_step_reward, \
    look_for_rewards_in_tree


def do_heuristic_model_learning(model: Dict,
                                Q: pd.DataFrame,
                                env,
                                gamma: float,
                                alpha: float,
                                depth: int,
                                start: int,
                                goal: int,
                                moves_made: int) -> None:
    """
    (f) from DYNA-H algorithm
    Allows to add heuristic in action selection

    :type Q: pandas DataFrame
    :param moves_made:
    :param goal:
    :param start:
    :param depth:
    :param alpha:
    :param gamma:
    :param model: dictionary representing a model
    :param Q: q values
    :param n: number of times model learning should be performed
    :param env: environment
    """
    s = env.state
    a = get_action_using_lookahead_heuristic(state=s,
                                             depth=depth,
                                             start=start,
                                             goal=goal,
                                             moves_made=moves_made)
    
    if a is None:
        previous_s_a = model.keys()
        s, a = random.choice(previous_s_a)
    
    result = model.get((s, a))  # obtain r and s' from the model
    
    # If action is not in the model, imagine it and update model
    if result is None:
        r = calculate_step_reward(action=a,
                                  goal_state=goal,
                                  start_position=start,
                                  moves_made=moves_made + 1,
                                  state=s)
        n_s = a
        update_model(model=model,
                     state=s,
                     action=a,
                     reward=r,
                     next_state=n_s)
    
    else:
        r, n_s = result
    
    q_value_update(Q=Q, s=s, a=a, r=r, s_prime=n_s, gamma=gamma, alpha=alpha)


def get_planned_rewards_using_lookahead(depth, state, start, goal,
                                        moves_made) -> List:
    """
    For all the actions that  are possible to be taken from the
    state gets the max look-ahead values for specified look-ahead
    depth
    :param depth:
    :param state:
    :param start:
    :param goal:
    :param moves_made:
    :return: List of next action values
    """
    added = set()
    added.add(state)
    
    actions = get_possible_actions(state)
    action_values = []
    
    for a in actions:
        
        if depth == 0:
            reward = calculate_step_reward(a, goal, start, moves_made, state)
            action_values.append(reward)
        else:
            action_values.append(
                look_ahead_for_max_rewards(depth=depth - 1,
                                           state=a,
                                           start=start,
                                           goal=goal,
                                           moves_made=moves_made + 1,
                                           added=added)
            )
    return action_values


def look_ahead_for_max_rewards(depth: int,
                               state: int,
                               start: int,
                               goal: int,
                               moves_made: int,
                               added: Set) -> float:
    """
    Each call to this function simulates imagined move made and observing
    the reward that was returned by that action.
    :param depth:
    :param state:
    :param start:
    :param goal:
    :param moves_made:
    :return:
    """
    
    added.add(state)
    
    actions = get_possible_actions(state)
    action_values = []
    
    for a in actions:
        
        if a in added:
            continue
        
        if depth == 0:
            reward = calculate_step_reward(action=a,
                                           goal_state=goal,
                                           start_position=start,
                                           moves_made=moves_made,
                                           state=state)
            action_values.append(reward)
        else:
            action_values.append(
                look_ahead_for_max_rewards(depth=depth - 1,
                                           state=a,
                                           start=start,
                                           goal=goal,
                                           moves_made=moves_made + 1,
                                           added=added)
            )
    return max(action_values)


def get_action_using_lookahead_heuristic(state: int,
                                         depth: int,
                                         start: int,
                                         goal: int,
                                         moves_made: int) -> int:
    """
    Finds the best action that can be taken in the state. This action is
    found using look-ahead tree. Algorithm finds the maximum reward that
    can be obtained after specified lookahead.
    :param state:
    :param depth: represent how many moves ahead algorithm looks for the
    rewards.
    :param start:
    :param goal:
    :param moves_made:
    :return: Returns the best of immediate actions with look-ahead
    depth
    """
    
    actions = get_possible_actions(state)
    # action_values = get_planned_rewards_using_lookahead(depth=depth,
    #                                                     state=state,
    #                                                     start=start,
    #                                                     goal=goal,
    #                                                     moves_made=moves_made)
    
    action_values = look_for_rewards_in_tree(state=state,
                                             depth=depth,
                                             start=start,
                                             goal=goal,
                                             moves_made=moves_made)
    max_value = max(action_values)
    value_check = []
    
    for a, v in zip(actions, action_values):
        if v == max_value:
            value_check.append(a)
    
    if len(value_check) == 0:
        return None
    # If only 1 action, it is chosen
    return random.choice(value_check)


if __name__ == '__main__':
    """
    get_action_using_lookahead_heuristic(state: int,
                                         depth: int,
                                         start: int,
                                         goal: int,
                                         moves_made: int)
    """
    a = get_action_using_lookahead_heuristic(33, )
