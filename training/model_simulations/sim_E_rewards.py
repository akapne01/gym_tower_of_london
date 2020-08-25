import time

from training.model_simulations.rewards_lookahead import dyna_rewards_lookahead
from training.utils.parameters import epsilon, transition_times, \
    epsilon_decay, version, alpha, gamma, min_epsilon

"""
Experimental Problem E
"""
start = 23
goal = 62

letter = 'E'

episodes = 1000

participants = 40

if __name__ == '__main__':
    time_per_depth = []
    start_time = time.time()

    for pid in range(50, 90):

        for i in range(0, 5):
            depth = i
            start_time_per_depth = time.time()
            dyna_rewards_lookahead(alpha=alpha,
                                   gamma=gamma,
                                   epsilon=epsilon,
                                   epsilon_decay=epsilon_decay,
                                   min_epsilon=min_epsilon,
                                   transition_times=transition_times,
                                   episodes=episodes,
                                   depth=depth,
                                   render=False,
                                   start=start,
                                   goal=goal,
                                   letter=letter,
                                   env_version=version,
                                   pid=pid)
            run_time = time.time() - start_time_per_depth
            time_per_depth.append(run_time)
            print(time_per_depth)
    print("--- TOTAL TIME:  %s seconds ---" % (time.time() - start_time))
