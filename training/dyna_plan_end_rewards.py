import random

import gym
from networkx.tests.test_convert_pandas import pd

from training.dyna_planning import save_stats_csv, reset_episode, initialise, \
    greedy_planning_look_ahead, \
    update_model
import numpy as np

from training.utils.planning_helper import get_possible_actions, init_q_table, \
    get_best_Q
from training.utils.tree_methods import search_tree, search_tree_end_rewards, \
    planning_algorithm, planning_algorithm_2, planning_algorithm_new


def model_learning_with_heuristic_new(model, Q, env, gamma, alpha, depth,
                                      start, goal, moves_made) -> None:
    """
    (f) from DYNA-H algorithm
    Allows to add heuristic in action selection

    :param model: dictionary representing a model
    :param Q: q values
    :param n: number of times model learning should be performed
    :param env: environment
    """
    s = env.state
    # Wors action selected
    a = planning_algorithm_new(s, depth, start, goal, moves_made)
    print(f'Planning algorithm returns a={a}')
    # If worst action not found then action is selected randomly from state
    # and actions taken previously
    if a is None:
        print(f'# model_learning_with_heuristic: returned action=None. SELECTING RANDOMLY from  next actions')
        previous_s_a = model.keys()
        s, a = random.choice(previous_s_a)

    # if state and action is not in the model then random action and state selected prev taken
    result = model.get((s, a))  # obtain r and s' from the model

    # If action is not in the model
    if result is None:
        print(
            f'# model_learning_with_heuristic: Action was not in the model, select randomly keys from model')
        (s, a), (r, n_s) = random.choice(list(model.items()))
    else:
        print(
            f'# model_learning_with_heuristic: Result={result} PROCEEDING WITH HEURISTIC VALUES')
        r, n_s = result
    print(f'# model_learning_with_heuristic: (s, a), (r, n_s)')
    print(f'# model_learning_with_heuristic: ({s}, {a}), ({r}, {n_s})')
    # q_learning_update2(Q, s, a, r, n_s, env, tree, gamma, alpha)
    q_learning_update2(Q, s, a, r, n_s, env, gamma, alpha)


# def perform_model_learning2(model, Q, n, env, tree, gamma, alpha) -> None:
def model_learning_with_heuristic(model, Q, n, env, gamma, alpha, tree, depth,
                                  start, goal, moves_made) -> None:
    """
    (f) from DYNA-H algorithm
    Allows to add heuristic in action selection

    :param model: dictionary representing a model
    :param Q: q values
    :param n: number of times model learning should be performed
    :param env: environment
    """
    for i in range(n):
        # add action selection based on heuristic
        s = env.state
        # a = greedy_planning_look_ahead(Q, env, s, tree)
        # a = planning_algorithm(model, s, depth, Q)
        a = planning_algorithm_2(s, depth, start, goal, moves_made)
        if a is None:
            print(
                f'# model_learning_with_heuristic: returned action=None. SELECTING RANDOMLY from  next actions')
            a = random.choice(get_possible_actions(s))
        # if state and action is not in the model then random action and state selected prev taken
        result = model.get((s, a))
        if result is None:
            print(
                f'# model_learning_with_heuristic: Result was None. SELECTING RANDOMLY')
            (s, a), (r, n_s) = random.choice(list(model.items()))
        else:
            print(
                f'# model_learning_with_heuristic: Result={result} PROCEEDING WITH HEURISTIC VALUES')
            r, n_s = result
        print(f'# model_learning_with_heuristic: (s, a), (r, n_s)')
        print(f'# model_learning_with_heuristic: ({s}, {a}), ({r}, {n_s})')
        # q_learning_update2(Q, s, a, r, n_s, env, tree, gamma, alpha)
        q_learning_update2(Q, s, a, r, n_s, env, gamma, alpha)


# def q_learning_update2(Q, s, a, r, s_prime, env, tree, gamma, alpha):
def q_learning_update2(Q, s, a, r, s_prime, env, gamma, alpha):
    """
    TD Update
    """
    print(f'# q_learning_update2 state={s} and next state is {s_prime}')
    best_next_actions = get_best_Q(s_prime, Q)
    print(f'# q_learning_update2: Best NEXT actions: {best_next_actions}')
    for action in best_next_actions:
        print(f' # (FOR) q_learning_update2: action={action}')
        # best_next_action = greedy_planning_look_ahead(Q, env, s_prime, tree)
        print(
            f' # (FOR) q_learning_update2: Q.loc[action, s_prime]={Q.loc[action, s_prime]}')
        print(f' # (FOR) q_learning_update2: Q.loc[a, s]={Q.loc[a, s]}')
        td_target = r + gamma * Q.loc[action, s_prime]
        td_delta = td_target - Q.loc[a, s]
        print(
            f' # (FOR) q_learning_update2: td_target={td_target}, td_delta={td_delta}, Q.loc[a, s] value now is = {Q.loc[a, s] + alpha * td_delta}')
        Q.loc[a, s] = Q.loc[a, s] + alpha * td_delta


#
# def get_action_max_q_value(q, s):
#     columns = get_possible_actions(s)
#     columns.append(s)
#     print(f'# get_action_max_q_value State: {s}')
#     print(f'# get_action_max_q_value Columns: {columns}')
#     # print(f'Actions: {actions}')
#     max = q[s].max()
#     df = q[columns]
#     print(df)
#     df = df.loc[df[s] == max]
#     indexes = df.index
#     poss_actions_max = []
#     for action in get_possible_actions(s):
#         if action in indexes:
#             poss_actions_max.append(action)
#     print(f'# get_action_max_q_value :  {poss_actions_max}')
#     if len(poss_actions_max) == 1:
#         return poss_actions_max[0]
#     return random.choice(poss_actions_max)


def take_action_using_epsilon_greedy(state, q, epsilon):
    """

    :param env: environment instance
    :param s: State
    :param q: q_table
    :return: action
    """
    actions = get_possible_actions(state)

    if np.random.rand() < epsilon:
        a = random.choice(actions)
        print(
            f'# take_action_using_epsilon_greedy - RANDOM ACTION SELECTED: {a}')
        return a
    print(
        f'# take_action_using_epsilon_greedy - Getting action to return by policy')
    a = get_best_Q(state, q)
    print(
        f'# take_action_using_epsilon_greedy - Max Q selected returning: {a}')
    if len(a) == 1:
        return a[0]
    return random.choice(a)


def dyna_lookahead_heuristic(alpha: float, gamma: float, epsilon: float,
                             episodes: int,
                             max_steps: int, depth: int,
                             render: bool, start: int, goal: int,
                             transition_times=1,
                             letter='NA', version='v0', epsilon_decay=0.999,
                             pid=1):
    """
    Dyna Algorithm with possibility to look-ahead to simulate planning
    :param version: v0 - give rewards on every step, v1 - returns reward after episode is complete
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

    env = gym.make(f'TolTask-{version}', start_state=start, goal_state=goal)
    env.delay = 0
    pid_results = []
    Q, window, episode_reward, episode_moves, moves_counter, model, move_path_monitor, actions_taken, epsilon_monitor = initialise()
    """
    Look forever
        Q, window, episode_reward, episode_moves, moves_counter, model, move_path_monitor, actions_taken, epsilon_monitor = initialise()

    """
    for episode in range(episodes):
        print(f"### Episode: {episode} ###")
        t, total_return, path, action_numbers_monitor = reset_episode()
        s = env.reset()
        tree = search_tree_end_rewards(s, depth, s, env.goal_state,
                                       env.counter)

        # a = greedy_planning_look_ahead(Q, env, s, tree)
        print('~~ a = take_action_using_epsilon_greedy(s, Q, epsilon) ~~ ')
        a = take_action_using_epsilon_greedy(s, Q, epsilon)

        while t < max_steps:

            if render:
                env.render()
            t += 1

            """
            (c) Step, observe R, S'
            """
            print(f'# Take step in env action={a}')
            s_prime, reward, done, info = env.step(a)
            action_numbers_monitor.append(a)

            action_value = info.get('action')
            path.append(action_value)
            # total_return += np.power(gamma, t) * reward
            total_return += reward
            # a_prime = greedy_planning_look_ahead(Q, env, s_prime, tree)

            """
            (d) Update Q(s, a) 
            """
            # q_learning_update2(Q, s, a, reward, s_prime, env, tree, gamma, alpha)
            print(' ~~ Perform update ~~')
            q_learning_update2(Q, s, a, reward, s_prime, env, gamma, alpha)

            print(' ~~ Perform MODEL update ~~')
            model = update_model(model, s, a, reward, s_prime)
            # perform_model_learning2(model, Q, transition_times, env, tree, gamma, alpha)
            print(' ~~ MODEL LEARNING and update ~~')
            model_learning_with_heuristic(model, Q, transition_times, env,
                                          gamma, alpha, tree, depth,
                                          env.initial_state,
                                          env.goal_state, env.counter)
            print(f'~~~ selecting a_prime ~~~')
            a_prime = take_action_using_epsilon_greedy(s_prime, Q, epsilon)

            s = s_prime
            a = a_prime

            epsilon = epsilon * epsilon_decay

            if done:
                if render:
                    print(
                        f"This episode took {t} timesteps and reward: {total_return}")

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
                epsilon_monitor.append(epsilon)
                print(f'~~~ MODEL: {model}')

                if verbose:
                    print(
                        "Episode: {0}\t Num_Steps: {1:>4}\tTotal_Return: {2:>5.2f}\tFinal_Reward: {3}\tEpsilon: {4:.3f}\tSuccess Rate: {5:.3f}\tLast_100: {6}".format(
                            episode, t, total_return, reward, epsilon,
                            moves_counter,
                            last_100))

                break
    print('Timestep_Reward:')
    print(episode_reward)
    print('number_of_moves')
    print(episode_moves)
    print('Epsilon values')
    stats = save_stats_csv(Q, letter, episodes, alpha, gamma, epsilon, depth,
                           episode_moves, episode_reward, actions_taken,
                           move_path_monitor, transition_times, version,
                           env.min_moves, epsilon_monitor, pid)
    # pid_results.append(stats)
    print('Training Complete')
    return epsilon_monitor, stats


def dyna_lookahead_heuristic_new(alpha: float, gamma: float, epsilon: float,
                                 episodes: int,
                                 max_steps: int, depth: int,
                                 render: bool, start: int, goal: int,
                                 transition_times=1,
                                 letter='NA', version='v0',
                                 epsilon_decay=0.999, pid=999,
                                 min_epsilon = 0.3):
    """
    Dyna Algorithm with possibility to look-ahead to simulate planning
    :param version: v0 - give rewards on every step, v1 - returns reward after episode is complete
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

    env = gym.make(f'TolTask-{version}', start_state=start, goal_state=goal)
    env.delay = 0
    Q, window, episode_reward, episode_moves, moves_counter, model, move_path_monitor, actions_taken, epsilon_monitor = initialise()
    """
    Loop through the episodes
    """

    # Trial
    for episode in range(episodes):
        # Begining of an episode
        a = None
        print(f"### Episode: {episode} ###")
        t, total_return, path, action_numbers_monitor = reset_episode()
        s = env.reset()
        not_complete = True
        # Time-step Loop
        while not_complete:
            a = take_action_using_epsilon_greedy(s, Q, epsilon)
            s_prime, reward, done, info = env.step(a)

            if render:
                env.render()

            model = update_model(model, s, a, reward, s_prime)
            q_learning_update2(Q, s, a, reward, s_prime, env, gamma, alpha)

            print(f'~~~ MODEL: {model}')

            # planning module
            for i in range(transition_times):
                model_learning_with_heuristic_new(model, Q, env, gamma, alpha,
                                                  depth,
                                                  start, goal, env.counter)
                print(f'~~~ selecting a_prime ~~~')

                s = s_prime

                # Epsilon is in the range (min_epsilon, epsilon]
                # After reaching minimum is not reduced further
                if epsilon > min_epsilon:
                    epsilon = epsilon * epsilon_decay

            # Add time step statistics
            total_return += reward
            action_numbers_monitor.append(a)

            action_value = info.get('action')
            path.append(action_value)

            if done:
                not_complete = False

                """
                Add episode statistics
                """
                move_count = info.get('count')
                moves_counter.append(move_count)
        # last_100 = window.count(1)

        # Add statistics
        episode_reward.append(total_return)
        actions_taken.append(action_numbers_monitor)
        move_path_monitor.append(path)
        epsilon_monitor.append(epsilon)
    # Before end of episode after game was finished
    save_stats_csv(Q, letter, episodes, alpha, gamma, epsilon, depth,
                   moves_counter, episode_reward, actions_taken,
                   move_path_monitor, transition_times, version,
                   env.min_moves, epsilon_monitor, pid)
    print('Training Complete')
