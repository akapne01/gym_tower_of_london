from typing import List

import numpy as np
from networkx.tests.test_convert_pandas import pd

from envs.custom_tol_env_dir import ToLTaskEnv
from envs.custom_tol_env_dir.tol_2d.mapping import int_to_state, state_to_int
from envs.custom_tol_env_dir.tol_2d.state import TolState

FILL_VALUE = -100


def init_q_table() -> pd.DataFrame:
    """
    Initializes q_value table. State is represented by columns,
    actions that can be taken in this state is represented as
    row. Actions that are not possible to be taken are initialized
    to FILL_VALUE = -100
    """
    poss_states = 36
    poss_actions = 36
    
    q_table = pd.DataFrame(
        np.array([[FILL_VALUE] * poss_actions] * poss_states))
    
    int_states = int_to_state.keys()  # contains all possible state numbers
    
    q_table.index = int_states
    q_table.columns = int_states
    
    for i in int_states:
        actions = get_possible_actions(i)
        for a in actions:
            q_table.loc[a, i] = 0
    
    return q_table


def get_best_Q_value(state: int, Q: pd.DataFrame) -> float:
    """
    Gets the next actions and find the
    max of the action Q values for specified
    state. Returns max Q-value
    """
    actions = get_possible_actions(state)
    df = Q.loc[actions]
    max_s = df[state].max()
    return max_s


def get_a_with_max_q(state: int, Q: pd.DataFrame) -> List:
    """
    Gets the next actions and find the
    max of the action Q values for specified
    state. Returns list of actions that have
    max q_values from specified state.
    """
    best = []
    actions = get_possible_actions(state)
    
    df = Q.loc[actions]  # slices df to only possible actions
    max_s = df[state].max()
    for a in actions:
        value = Q.loc[a, state]
        if value == max_s:
            best.append(a)
    return best


def get_min_q(state: int, Q: pd.DataFrame) -> List:
    """
    Gets the next actions and find the
    max of the action Q values for specified
    state. Returns list of actions that have
    max q_values from specified state.
    """
    worst = []
    actions = get_possible_actions(state)

    df = Q.loc[actions]  # slices df to only possible actions
    min_s = df[state].min()
    for a in actions:
        value = Q.loc[a, state]
        if value == min_s:
            worst.append(a)
    return worst


def get_possible_actions(state: int) -> List:
    """
    Calculates which actions can be taken from specified state.
    Returns a list of action numbers.
    Length of this list can vary from 2 actions minimum to
    4 actions maximum that can be taken from state.
    """
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
