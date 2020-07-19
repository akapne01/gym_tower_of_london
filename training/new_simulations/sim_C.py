import time
import random
from training.dyna_plan_end_rewards import dyna_lookahead_heuristic_new
from training.run_simulations.parameters import alpha, gamma, epsilon, \
    max_steps, transition_times, epsilon_decay

"""
Experimental Problem C
"""

start = 24
goal = 66

letter = 'C'

episodes = 700

participants = 2

time_per_depth = []
start_time = time.time()

for pid in range(participants):
    # Randomly sample alpha and gamma to simulate
    # different participants
    alpha = random.uniform(0, 1)
    gamma = random.uniform(0, 1)
    # Use different lookahead depths
    # Depth 0 looks 1 move ahead
    # Depth 1 loooks 2 moves ahead, etc.
    for i in range(4):
        depth = i
        start_time_per_depth = time.time()
        dyna_lookahead_heuristic_new(alpha,
                                     gamma,
                                     epsilon,
                                     episodes,
                                     max_steps,
                                     depth,
                                     pid=pid,
                                     render=False,
                                     start=start,
                                     goal=goal,
                                     letter=letter,
                                     transition_times=transition_times,
                                     version='v0',
                                     epsilon_decay=epsilon_decay)
        run_time = time.time() - start_time_per_depth
        time_per_depth.append(run_time)
        print(time_per_depth)
print("--- TOTAL TIME:  %s seconds ---" % (time.time() - start_time))
