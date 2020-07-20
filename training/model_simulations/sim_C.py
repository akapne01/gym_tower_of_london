import time
import random
from training.dyna_plan_end_rewards import dyna_heuristic_with_lookahead
from training.utils.parameters import epsilon, transition_times, \
    epsilon_decay, version, min_epsilon, alpha, gamma

"""
Experimental Problem C
"""

start = 24
goal = 66

letter = 'C'

episodes = 1000

participants = 2

time_per_depth = []
start_time = time.time()

for pid in range(participants):
    
    # alpha = random.uniform(0, 1)
    # gamma = random.uniform(0, 1)
    
    for i in range(0, 5):
        depth = i
        start_time_per_depth = time.time()
        dyna_heuristic_with_lookahead(alpha=alpha,
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
