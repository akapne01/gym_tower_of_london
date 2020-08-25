import random
from typing import Dict

import gym
import pandas as pd
from training.utils.dyna_helper_methods import save_stats_csv, reset_episode, \
    initialise_algorithm, \
    update_model, select_action_epsilon_greedy, q_value_update, select_action_with_min_q, \
    select_action_with_greedy_policy

from training.utils.parameters import LAST_MOVES_COUNT
from training.utils.planning_helper import get_possible_actions
from training.utils.tree_methods import calculate_step_reward, \
    look_for_rewards_in_tree


def get_action_using_lookahead_heuristic(state: int,
                                         depth: int,
                                         start: int,
                                         goal: int,
                                         moves_made: int,
                                         q_values: pd.DataFrame) -> int:
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

    # action_values = look_for_q_values_in_tree(state=state,
    #                                           depth=depth,
    #                                           start=start,
    #                                           goal=goal,
    #                                           moves_made=moves_made,
    #                                           q_values=q_values)
    # max_value = max(action_values)
    min_value = min(action_values)
    value_check = []

    for a, v in zip(actions, action_values):
        # if v == max_value:
        if v == min_value:
            value_check.append(a)

    if len(value_check) == 0:
        return None
    # If only 1 action, it is chosen
    return random.choice(value_check)


def do_heuristic_model_learning(model: Dict,
                                Q: pd.DataFrame,
                                env,
                                gamma: float,
                                alpha: float,
                                depth: int,
                                start: int,
                                goal: int,
                                moves_made: int,
                                epsilon: float) -> None:
    """
    (f) from DYNA-H algorithm
    Allows to add heuristic in action selection

    :param epsilon:
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
    if depth == 0:
        # a = select_action_with_greedy_policy(state=s, q=Q)
        a = select_action_epsilon_greedy(state=s, q=Q, epsilon=epsilon)
        # a = select_action_with_min_q(state=s, q=Q)
    else:
        a = get_action_using_lookahead_heuristic(state=s,
                                                 depth=depth,
                                                 start=start,
                                                 goal=goal,
                                                 moves_made=moves_made,
                                                 q_values=Q)

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


def dyna_rewards_lookahead(alpha: float,
                           gamma: float,
                           epsilon: float,
                           episodes: int,
                           depth: int,
                           render: bool,
                           start: int,
                           goal: int,
                           transition_times: int,
                           letter: str,
                           env_version: str,
                           epsilon_decay: float,
                           min_epsilon: float,
                           pid=999) -> None:
    """
    Dyna Algorithm with possibility to look-ahead to simulate planning
    :param pid:
    :param min_epsilon:
    :param epsilon_decay:
    :param env_version: v0 - give rewards on every step, v1 - returns reward after
    episode is complete
    :param alpha: step-size parameter
    :param gamma: discount-rate parameter
    :param epsilon: probability of taking a random action
    :param episodes: NUmber of trials
    :param depth: Number of moves to lookahead for rewards
    :param render: Boolean representing if GUI should be rendered
    :param start: ToL start state
    :param goal: ToL goal state
    :param transition_times: Represents the number of times model is used to
    update Q-values in each trial
    :param letter: letter denoting the experimental problem. Used when saving
    statistics
    """

    env = gym.make(f'TolTask-{env_version}', start_state=start, goal_state=goal)
    env.delay = 0
    Q, window, episode_reward, episode_moves, \
    moves_counter, model, move_path_monitor, \
    actions_taken, epsilon_monitor = initialise_algorithm()
    """
    Loop through the episodes
    """

    # Trial
    for episode in range(episodes):

        # Initialize episode
        a = None
        print(f"### Episode: {episode} ###")
        t, total_return, path, action_numbers_monitor = reset_episode()
        s = env.reset()
        not_complete = True

        # Time-step Loop
        while not_complete:

            a = select_action_epsilon_greedy(state=s, q=Q, epsilon=epsilon)

            s_prime, reward, done, info = env.step(a)

            if render:
                env.render()

            model = update_model(model=model,
                                 state=s,
                                 action=a,
                                 reward=reward,
                                 next_state=s_prime)

            q_value_update(Q=Q,
                           s=s,
                           a=a,
                           r=reward,
                           s_prime=s_prime,
                           gamma=gamma,
                           alpha=alpha)

            # planning module
            for i in range(transition_times):
                do_heuristic_model_learning(model=model,
                                            Q=Q,
                                            env=env,
                                            gamma=gamma,
                                            alpha=alpha,
                                            depth=depth,
                                            start=start,
                                            goal=goal,
                                            moves_made=env.counter,
                                            epsilon=epsilon)

            s = s_prime

            # Add time step statistics
            total_return += reward
            action_numbers_monitor.append(a)
            action_value = info.get('action')
            path.append(action_value)

            if done:
                not_complete = False

                # Add episode statistics after finished
                move_count = info.get('count')
                moves_counter.append(move_count)

        if epsilon > min_epsilon:
            epsilon = epsilon * epsilon_decay

        # Add episode statistics
        episode_reward.append(total_return)
        actions_taken.append(action_numbers_monitor)
        move_path_monitor.append(path)
        epsilon_monitor.append(epsilon)

    # After all episodes finished
    save_stats_csv(Q=Q,
                   letter=letter,
                   episodes=episodes,
                   alpha=alpha,
                   gamma=gamma,
                   epsilon=epsilon,
                   depth=depth,
                   episode_moves=moves_counter,
                   episode_reward=episode_reward,
                   actions_taken=actions_taken,
                   move_path_monitor=move_path_monitor,
                   transition_times=transition_times,
                   version=env_version,
                   min_moves=env.min_moves,
                   epsilon_monitor=epsilon_monitor,
                   pid=pid,
                   no_last_moves=LAST_MOVES_COUNT,
                   type='rewards')
    print('Training Complete')
