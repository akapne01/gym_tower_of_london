import time

from training.dyna_plan_end_rewards import dyna_lookahead_heuristic
from training.dyna_planning import dyna_with_lookahead
from training.run_simulations.parameters import alpha, gamma, epsilon, max_steps, transition_times

"""
Experimental Problem D
"""
start = 16
goal = 46
letter = 'D'

episodes = 10000
# lookahead_depths = [1, 2, 3, 4]
# lookahead_depths = [5, 6, 7]
lookahead_depths = [1]

time_per_depth = []
start_time = time.time()
for depth in lookahead_depths:
    start_time_per_depth = time.time()
    print('##################################################################################')
    print(f'Running EXPERIMENTAL_PROBLEM {letter} DYNA-LOOKAHEAD with lookahead-depth {depth}')
    print('##################################################################################')

    # dyna_with_lookahead(alpha, gamma, epsilon, episodes, max_steps, depth, render=False, start=start,
    #                     goal=goal, letter=letter, transition_times=transition_times)
    dyna_lookahead_heuristic(alpha, gamma, epsilon, episodes, max_steps, depth, render=False, start=start,
                             goal=goal, letter=letter, transition_times=transition_times, version='v0')
    run_time = time.time() - start_time_per_depth
    time_per_depth.append(run_time)
print("--- TOTAL TIME:  %s seconds ---" % (time.time() - start_time))
print(time_per_depth)