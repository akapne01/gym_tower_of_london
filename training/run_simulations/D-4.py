import time

from networkx.tests.test_convert_pandas import pd

from training.dyna_plan_end_rewards import dyna_lookahead_heuristic
from training.run_simulations.parameters import alpha, gamma, epsilon, max_steps, transition_times, epsilon_decay, LAST_MOVES_COUNT

"""
Experimental Problem D
"""
start = 16
goal = 46
letter = 'D'

episodes = 1000

depth = 4
participants = 43
time_per_depth = []
pid_results = []
start_time = time.time()
all_results = []
col_last_moves = f'LAST_{LAST_MOVES_COUNT}_MOVES'

for pid in range(participants):
    # for pid in range(2):
    print(f'Running simulation for participant number {pid}')
    # for depth in lookahead_depths:
    start_time_per_depth = time.time()
    print('##################################################################################')
    print(f'Running EXPERIMENTAL_PROBLEM {letter} DYNA-LOOKAHEAD with lookahead-depth {depth}')
    print('##################################################################################')

    epsilon_monitor, results = dyna_lookahead_heuristic(alpha, gamma, epsilon, episodes, max_steps, depth, pid=pid,
                                                        render=False,
                                                        start=start, goal=goal,
                                                        letter=letter, transition_times=transition_times,
                                                        version='v0',
                                                        epsilon_decay=epsilon_decay)

    run_time = time.time() - start_time_per_depth
    time_per_depth.append(run_time)

    # params = f'ep-{episodes}_al-{alpha}_gam-{gamma}_eps-{epsilon}_depth-{depth}_model_transitions-{transition_times}'
    #
    # col_last_moves = f'LAST_{LAST_MOVES_COUNT}_MOVES'

    # Saving Statistics - get all runs in one csv
    # summary = pd.DataFrame()
    # for i in range(len(results)):
    #     try:
    #         # print(f'SO stats={stats} of type {type(stats)}')
    #         summary[f'{i}_col_last_moves'] = results[col_last_moves]
    #     except KeyError:
    #         summary[f'MOVE_COUNT'] = results['MOVE_COUNT']
    #
    # # params = f'ep-{episodes}_al-{alpha}_gam-{gamma}_eps-{epsilon}_depth-{depth}_model_transitions-{transition_times}-pid={pid}'
    # summary.to_csv(f'STATS/summary_{letter}_statistics_{params}.csv')

    # all_results.append(summary)
    print(time_per_depth)

print("--- TOTAL TIME:  %s seconds ---" % (time.time() - start_time))
#

# After all run complete
# all = pd.DataFrame()
# for i, stats in enumerate(all_results):
#     print(f'i={i}, stats={stats}')
#     try:
#         all[f'{i}_col_last_moves'] = stats[col_last_moves]
#     except KeyError:
#         if stats.empty:
#             all[f'{i}_MOVE_COUNT'] = -1
#         else:
#             all[f'{i}_MOVE_COUNT'] = stats[f'MOVE_COUNT']
# params = f'ep-{episodes}_al-{alpha}_gam-{gamma}_eps-{epsilon}_depth-{depth}_model_transitions-{transition_times}'
# # all.to_csv(f'STATS/ALL_summary_{letter}_statistics_{params}.csv')
# all.to_csv(f'STATS/ALL_summary_total_{participants}_{letter}_statistics_{params}.csv')