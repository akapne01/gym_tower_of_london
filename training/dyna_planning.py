"""
Modified Tabular Dyna-Q with reward look-ahead
"""
import random
from collections import deque
from typing import Tuple, Dict

import gym
import numpy as np
from networkx.tests.test_convert_pandas import pd

from training.utils.planning_helper import init_q_table, get_best_Q
from training.utils.tree_methods import search_tree, plan_for_best_actions


def greedy_planning_look_ahead(Q, env, state, tree):
    """
    If state is in the tree then lookahead. If lookahead
    returns back more than one action then select the action
    with highest Q value.

    If state is not in the tree then select the action with
    highest Q values.
    If only one action then return it, if more than one action
    then return random action.

    @param Q Q values state x action -> value
    @param epsilon for exploration
    @param s number of states
    @param train if true then no random actions selected
    """
    print('# greedy_planning_look_ahead: Looking for the best action: ', state)
    if state in tree:
        actions = plan_for_best_actions(state, tree)
        if actions is None or len(actions) > 1:
            print('# greedy_planning_look_ahead: planning returned more than 1 action:', actions)
            print('# greedy_planning_look_ahead:Getting the best Q value bacause of that')
            actions = get_best_Q(state, Q)
    else:
        actions = get_best_Q(state, Q)
    if len(actions) == 0:
        a = env.get_random_action()
        print('# greedy_planning_look_ahead: List is empty so have to take random action', a)
        return a
    # So far we have a list of action that may contain 1 or more action
    if len(actions) == 1:
        print('#greedy_planning_look_ahead#: Only 1 best action was found, returning this action', actions[0])
        return actions[0]
    else:
        print('#greedy_planning_look_ahead#: There are several best actions', actions)
        action = random.choice(actions)
        print('#greedy_planning_look_ahead#: Randomly selecting one of the actions ', action)
        return action


def q_learning_update(Q, s, a, r, s_prime, env, tree, gamma, alpha):
    """
    TD Update
    """
    best_next_action = greedy_planning_look_ahead(Q, env, s_prime, tree)
    td_target = r + gamma * Q.loc[best_next_action, s_prime]
    td_delta = td_target - Q.loc[a, s]
    return Q.loc[a, s] + alpha * td_delta


def update_model(model, state, action, reward, next_state) -> Dict:
    """
    Stores visited states in the model.
    Model is represented as a dictionary
    """
    model[(state, action)] = reward, next_state
    return model


def perform_model_learning(model, Q, n, env, tree, gamma, alpha) -> None:
    """
    (f) from DYNA algorithm
    :param model: dictionary representing a model
    :param Q: q values
    :param n: number of times model learning should be performed
    :param env: environment
    """
    for i in range(n):
        (s, a), (r, n_s) = random.choice(list(model.items()))
        print(f'Model learning: (s, a), (r, n_s)')
        print(f'Model learning: ({s}, {a}), ({r}, {n_s})')
        q_learning_update(Q, s, a, r, n_s, env, tree, gamma, alpha)


def initialise() -> Tuple:
    """
    Initalize varaibles to record statistics to record statistics
    :return: Q, window, episode_reward, episode_moves, moves_counter,
            model, move_path_monitor, actions_taken
    """
    Q = init_q_table()
    window = deque(maxlen=100)
    model = dict()
    return Q, window, [], [], [], model, [], []


def save_stats_csv(Q, letter, episodes, alpha, gamma, epsilon, depth, episode_moves, episode_reward, actions_taken,
                   move_path_monitor, transition_times, version) -> None:
    """
    Saves 2 CSV files:
    Q_values and training statistics. Parameters are saved in the CSV name
    """
    q_value_path = f'{letter}/{letter}_{version}_DYNA_QVALUES_ep-{episodes}_al-{alpha}_gam-{gamma}_eps-{epsilon}_depth-{depth}_model_transitions-{transition_times}.csv'
    Q.to_csv(q_value_path)

    stats_path = f'{letter}/{letter}_{version}_DYNA_episode-{episodes}_alpha-{alpha}_gamma-{gamma}_epsilon-{epsilon}_depth-{depth}.csv'
    df = pd.DataFrame(data=episode_moves, columns=['MOVE_COUNT'])
    df['EPISODE_REWARDS'] = episode_reward
    df['ACTIONS_NUMBERS'] = actions_taken
    df['PATHS'] = move_path_monitor
    df.to_csv(stats_path, index=False)


def reset_episode():
    """
    Resets the stats to record episode events
    :return:  t, total_return, path, action_numbers_monitor
    """
    return 0, 0.0, [], []


def dyna_with_lookahead(alpha: float, gamma: float, epsilon: float, episodes: int, max_steps: int, depth: int,
                        render: bool, start: int, goal: int, transition_times=1, letter='NA', version='v0') -> None:
    """
    Dyna Algorithm with possibility to look-ahead to simulate planning
    :param version:
    :param alpha: step-size parameter
    :param gamma: discount-rate parameter
    :param epsilon: probability of taking a random action
    :param episodes: NUmber of trials
    :param max_steps: Represents after how many steps should terminate the algorithm and disregard the trial
    :param depth: Number of moves to lookahead for rewards
    :param render: Boolean representing if GUI should be rendered
    :param start: ToL start state
    :param goal: ToL goal state
    :param transition_times: Represents the number of times model is used to update Q-values in each trial
    :param letter: letter denoting the experimental problem. Used when saving statistics
    """
    verbose = False

    # env = gym.make('TolTask-v0', start_state=start, goal_state=goal)
    env = gym.make('TolTask-v1', start_state=start, goal_state=goal)
    env.delay = 0

    Q, window, episode_reward, episode_moves, moves_counter, model, move_path_monitor, actions_taken = initialise()
    """
    Look forever
    """
    for episode in range(episodes):
        print(f"### Episode: {episode} ###")
        t, total_return, path, action_numbers_monitor = reset_episode()
        s = env.reset()
        tree = search_tree(s, depth, s, env.goal_state, env.counter)

        a = greedy_planning_look_ahead(Q, env, s, tree)

        while t < max_steps:

            if render:
                env.render()
            t += 1

            """
            (c) Step, observe R, S'
            """
            s_prime, reward, done, info = env.step(a)
            action_numbers_monitor.append(a)

            action_value = info.get('action')
            path.append(action_value)
            # total_return += np.power(gamma, t) * reward
            total_return += reward
            a_prime = greedy_planning_look_ahead(Q, env, s_prime, tree)

            """
            (d) Update Q(s, a) 
            """
            Q.loc[a, s] = q_learning_update(Q, s, a, reward, s_prime, env, tree, gamma, alpha)
            model = update_model(model, s, a, reward, s_prime)
            perform_model_learning(model, Q, transition_times, env, tree, gamma, alpha)

            s = s_prime
            a = a_prime

            if done:
                if render:
                    print(f"This episode took {t} timesteps and reward: {total_return}")

                """
                Add episode statistics
                """
                episode_reward.append(total_return)
                move_count = info.get('count')
                moves_counter.append(move_count)
                episode_moves.append(move_count)
                move_path_monitor.append(path)
                actions_taken.append(action_numbers_monitor)
                last_100 = window.count(1)

                if verbose:
                    print(
                        "Episode: {0}\t Num_Steps: {1:>4}\tTotal_Return: {2:>5.2f}\tFinal_Reward: {3}\tEpsilon: {4:.3f}\tSuccess Rate: {5:.3f}\tLast_100: {6}".format(
                            episode, t, total_return, reward, epsilon, moves_counter,
                            last_100))

                break
    print('Timestep_Reward:')
    print(episode_reward)
    print('number_of_moves')
    print(episode_moves)
    print('Epsilon values')
    save_stats_csv(Q, letter, episodes, alpha, gamma, epsilon, depth, episode_moves, episode_reward, actions_taken,
                   move_path_monitor, transition_times, version)
    print('Training Complete')


if __name__ == "__main__":
    alpha = 0.5
    gamma = 0.999
    epsilon = 0.9
    episodes = 1
    max_steps = 500
    depth = 4
    transition_times = 1
    s = 33
    g = 54
    dyna_with_lookahead(alpha, gamma, epsilon, episodes, max_steps, depth, render=True, start=s, goal=g,
                        transition_times=1)
