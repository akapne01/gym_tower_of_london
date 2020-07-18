import time
from training.dyna_plan_end_rewards import dyna_lookahead_heuristic_new
from training.run_simulations.parameters import alpha, gamma, epsilon, \
    max_steps, transition_times, epsilon_decay

"""
Experimental Problem B
"""
start = 51
goal = 32
letter = 'B'

episodes = 500


participants = 1

time_per_depth = []
start_time = time.time()
for i in range(4):
    depth = i
    for pid in range(participants):
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
