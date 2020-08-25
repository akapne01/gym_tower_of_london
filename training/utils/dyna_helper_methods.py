import os
import random
from collections import deque
from typing import Tuple, Dict, List

import numpy as np
from networkx.tests.test_convert_pandas import pd

from training.utils.planning_helper import init_q_table, get_possible_actions, \
    get_a_with_max_q, get_best_Q_value


def q_value_update(Q: pd.DataFrame,
                   s: int,
                   a: int,
                   r: int,
                   s_prime: int,
                   gamma: float,
                   alpha: float) -> None:
    """
    TD Update
    :param Q: q_values
    :param s: state
    :param a: action
    :param r: reward
    :param s_prime: next state
    :param gamma:
    :param alpha:
    """
    max_next_q = get_best_Q_value(s_prime, Q)  # max Q.loc[action, s_prime]
    td_target = r + gamma * max_next_q
    td_delta = td_target - Q.loc[a, s]
    Q.loc[a, s] = Q.loc[a, s] + alpha * td_delta


def select_action_with_min_q(state: int, q: pd.DataFrame, ) -> int:
    a = get_a_with_max_q(state, q)
    return random.choice(a)


def select_action_with_greedy_policy(state: int,
                                     q: pd.DataFrame) -> int:
    """
    Action selection policy.
    Simulates a biased coin flip by obtaining a random number
    between 0 and 1.
    If random number is smaller than epsilon, then choose action
    randomly. If bigger than epsilon, then action is selected
    based on the max q_value.
    If more than one action has maximum q_value then one of those
    actions is randomly selected. If only 1 action has max q_value
    then that value is returned.
    :return: action to take next
    """
    a = get_a_with_max_q(state, q)
    return random.choice(a)


def select_action_epsilon_greedy(state: int,
                                 q: pd.DataFrame,
                                 epsilon: float) -> int:
    """
    Action selection policy.
    Simulates a biased coin flip by obtaining a random number
    between 0 and 1.
    If random number is smaller than epsilon, then choose action
    randomly. If bigger than epsilon, then action is selected
    based on the max q_value.
    If more than one action has maximum q_value then one of those
    actions is randomly selected. If only 1 action has max q_value
    then that value is returned.
    :return: action to take next
    """
    if np.random.rand() < epsilon:
        actions = get_possible_actions(state)
        a = random.choice(actions)
        return a
    a = get_a_with_max_q(state, q)
    return random.choice(a)


def update_model(model: Dict,
                 state: int,
                 action: int,
                 reward: float,
                 next_state: int) -> Dict:
    """
    Stores visited states, actions and their rewards and
    next state in the model dictionary.
    """
    model[(state, action)] = reward, next_state
    return model


def initialise_algorithm() -> Tuple:
    """
    Initialize variables to record statistics to record statistics
    :return: Q, window, episode_reward, episode_moves, moves_counter,
            model, move_path_monitor, actions_taken
    """
    Q = init_q_table()
    window = deque(maxlen=100)
    model = dict()
    return Q, window, [], [], [], model, [], [], []


def save_stats_csv(Q: pd.DataFrame,
                   letter: str,
                   episodes: int,
                   alpha: float,
                   gamma: float,
                   epsilon: float,
                   depth: int,
                   episode_moves: List,
                   episode_reward: List,
                   actions_taken: List,
                   move_path_monitor: List,
                   transition_times: int,
                   version: str,
                   min_moves: int,
                   epsilon_monitor: List,
                   pid: int,
                   no_last_moves: int,
                   type=None) -> None:
    """
    Saves CSV files:
    Q_values and training moves and statistics. Parameters are saved in the CSV name
    """
    if type:
        params = f'ep={episodes}_pid={pid}_al={alpha}_gam={gamma}__depth=' \
                 f'{depth}_eps={epsilon}_mtrans={transition_times}_{type}'
    else:
        params = f'ep={episodes}_pid={pid}_al={alpha}_gam={gamma}__depth={depth}_eps={epsilon}_mtrans={transition_times}'
    q_value_path = f'{letter}/{letter}_{version}_dyna-h_qvalues_{params}.csv'
    stats_path = f'{letter}/{letter}_{version}_dyna-h_stats_{params}.csv'

    Q.to_csv(q_value_path)

    df = pd.DataFrame(data=episode_moves, columns=['MOVE_COUNT'])
    df['EPISODE_REWARDS'] = episode_reward
    df['ACTIONS_NUMBERS'] = actions_taken
    df['PATHS'] = move_path_monitor
    df['EPSILON'] = epsilon_monitor
    df['ALPHA'] = alpha
    df['GAMMA'] = gamma
    df.to_csv(stats_path, index=False)

    stats = pd.DataFrame(df.MOVE_COUNT.describe())
    stats.loc['no_min_moves', 'MOVE_COUNT'] = np.sum(
        df.MOVE_COUNT == min_moves)
    if len(df.MOVE_COUNT) > no_last_moves:
        df_last = df.tail(no_last_moves)
        col = f'LAST_{no_last_moves}_MOVES'
        stats[col] = df_last.MOVE_COUNT.describe()
        stats.loc['no_min_moves', col] = np.sum(
            df_last.MOVE_COUNT == min_moves)
    if type:
        dir = f'STATS/{letter}/{type}/depth_{depth}/'
    else:
        dir = f'STATS/{letter}/depth_{depth}/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    file = os.path.join(dir, f'{letter}_stats_{params}.csv')
    stats.to_csv(file)
    return stats


def reset_episode():
    """
    Resets the stats to record episode events
    :return:  t, total_return, path, action_numbers_monitor
    """
    return 0, 0.0, [], []
