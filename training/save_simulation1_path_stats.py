from os import listdir
from os.path import isfile, join

import pandas as pd


def read_data_in_df(files):
    df = pd.DataFrame()
    for f in files:
        print(f)
        frame = pd.read_csv(f)
        pid = f.find('pid=')
        col = f[pid:pid + 6]
        df[f'{col}_MOVE_COUNT'] = frame.MOVE_COUNT.tail(100)
        df[f'{col}_ACTIONS_NUMBERS'] = frame.ACTIONS_NUMBERS.tail(100)
        df[f'{col}_PATHS'] = frame.PATHS.tail(100)
    return df


def get_depth(name):
    n = name.find('depth=')
    start = n + 6
    d = name[start: start + 1]
    return int(d)


def get_files_in_dir(directory, name, depth):
    """
    Returns list of file name in directory that
    starts with specified name
    """
    return [join(directory, f) for f in listdir(directory) if isfile(join(
        directory, f)) and f.startswith(name) and get_depth(f) == depth]


if __name__ == '__main__':
    directory = '/Users/agnese/Dropbox/PyCharmProjects/gym_tower_of_london/training/model_simulations/{}/'
    name = '{}_v0_dyna-h_stats_ep=1000_pid='
    
    dir_a = directory.format('A')
    dir_b = directory.format('B')
    dir_c = directory.format('C')
    dir_d = directory.format('D')
    dir_e = directory.format('E')
    
    dir_A0 = get_files_in_dir(dir_a, name.format('A'), 0)
    dir_B0 = get_files_in_dir(dir_b, name.format('B'), 0)
    dir_C0 = get_files_in_dir(dir_c, name.format('C'), 0)
    dir_D0 = get_files_in_dir(dir_d, name.format('D'), 0)
    dir_E0 = get_files_in_dir(dir_e, name.format('E'), 0)
    
    a0 = read_data_in_df(dir_A0)
    b0 = read_data_in_df(dir_B0)
    c0 = read_data_in_df(dir_C0)
    d0 = read_data_in_df(dir_D0)
    e0 = read_data_in_df(dir_E0)
    
    a0.to_csv('simulation_1_moves_A0.csv', index=False)
    b0.to_csv('simulation_1_moves_B0.csv', index=False)
    c0.to_csv('simulation_1_moves_C0.csv', index=False)
    d0.to_csv('simulation_1_moves_D0.csv', index=False)
    e0.to_csv('simulation_1_moves_E0.csv', index=False)
