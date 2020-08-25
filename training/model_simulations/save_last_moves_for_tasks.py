from os import listdir
from os.path import isfile, join
from typing import Dict, List, Tuple

import pandas as pd

episodes = 1000
letters = ['A', 'B', 'C', 'D', 'E']

get_min = {
    'A': 4,
    'B': 5,
    'C': 5,
    'D': 6,
    'E': 6,
}


def sort_files_by_depth(files: List) -> Tuple[List, List, List, List, List]:
    """
    Sorts files by depth in separate list
    """
    depth_0 = []
    depth_1 = []
    depth_2 = []
    depth_3 = []
    depth_4 = []
    
    for f in files:
        d0 = f.find('depth=')
        d = f[d0 + 6: d0 + 7]  # extracts depth from file name
        if d == '0':
            depth_0.append(f)
        elif d == '1':
            depth_1.append(f)
        elif d == '2':
            depth_2.append(f)
        elif d == '3':
            depth_3.append(f)
        else:
            depth_4.append(f)
    return depth_0, depth_1, depth_2, depth_3, depth_4


def read_files(files: List) -> Dict:
    """
    Reads list of file names into dictionary
    :param files: list of file names
    :return: Dict where key is a file name and
    value is a pandas dataframe
    """
    dfs = {}
    for f in files:
        # print('Reading', f)
        df = pd.read_csv(f)
        dfs[f] = df
    return dfs


def get_file_dictionaries(d0, d1, d2, d3, d4) -> Tuple[Dict, Dict, Dict, Dict,
                                                       Dict]:
    """
    Reads files into dictionaries
    """
    dfs_0 = read_files(d0)
    dfs_1 = read_files(d1)
    dfs_2 = read_files(d2)
    dfs_3 = read_files(d3)
    dfs_4 = read_files(d4)
    return dfs_0, dfs_1, dfs_2, dfs_3, dfs_4


def combine_results(dfs: Dict) -> pd.DataFrame:
    """
    Combines all DataFrames in dictionary into
    one pd.DataFrame
    """
    frame = pd.DataFrame()
    for n, df in dfs.items():
        print('IN COMBINE results')
        print(n)
        # print(df)
        s = n.find('pid=')
        print(n[s + 4 : s + 6])
        # p = int(n[117:118])
        p = n[s + 4 : s + 6]
        print('pid', p)
        frame[p] = df.MOVE_COUNT.tail(100)
    return frame.reset_index(drop=True)


def get_combined_results(dfs_0: Dict,
                         dfs_1: Dict,
                         dfs_2: Dict,
                         dfs_3: Dict,
                         dfs_4: Dict) -> Tuple:
    """
    Wraps combine results for all depths
    """
    df0 = combine_results(dfs_0)
    df1 = combine_results(dfs_1)
    df2 = combine_results(dfs_2)
    df3 = combine_results(dfs_3)
    df4 = combine_results(dfs_4)
    return df0, df1, df2, df3, df4


def save_last_move_statistics():
    """
    Reads outputs of simulation CSV files.
    Saves CSV files for each experimental
    problem separately.
    Tasks: A, B, C, D and E
    Depths = 0, 1, 2, 3, 4
    Saves files for each task and depth in
    separate files
    """
    for letter in letters:
        directory = f'/Users/agnese/Dropbox/PyCharmProjects/gym_tower_of_london/training/model_simulations/{letter}/'
        start = f'{letter}_v0_dyna-h_stats_ep={episodes}'
        files = [join(directory, f) for f in listdir(directory) if
                 isfile(join(directory, f)) and f.startswith(start)]
        
        depth_0, depth_1, depth_2, depth_3, depth_4 = sort_files_by_depth(files)
        # print('depth0 = ', len(depth_0))
        # print('depth1 = ', len(depth_1))
        # print('depth2 = ',len( depth_2))
        # print('depth3 = ', len(depth_3))
        # print('depth4 = ', len(depth_4))
        dfs_0, dfs_1, dfs_2, dfs_3, dfs_4 = get_file_dictionaries(depth_0,
                                                                  depth_1,
                                                                  depth_2,
                                                                  depth_3,
                                                                  depth_4)
        # print('dfs_1', dfs_1)
        # print('dfs_2', dfs_2)
        # print('dfs_3', dfs_3)
        # print('dfs_4', dfs_4)
        # print('dfs_0', dfs_0)
     
        df0, df1, df2, df3, df4 = get_combined_results(dfs_0, dfs_1, dfs_2,
                                                       dfs_3, dfs_4)
        
        print('df0=', df0.shape)
        # print('df1=', df1)
        # print('df2=', df2)
        # print('df3=', df3)
        # print('df4=', df4)
        
        results = [df0, df1, df2, df3, df4]
        # print('RESUKTS:', results)
        
        # Save results
        for i, f in enumerate(results):
            file_name = f'experimental_results/{letter}_depth' \
                        f'={i}_last_100_moves.csv'
            f.to_csv(file_name, index=False)


def combine_results_by_depth(dfs: Dict) -> pd.DataFrame:
    """
    Combines all DataFrames in dictionary into
    one pd.DataFrame
    """
    frame = pd.DataFrame()
    # dfa = pd.DataFrame()
    # dfb = pd.DataFrame()
    # dfc = pd.DataFrame()
    # dfd = pd.DataFrame()
    # dfe = pd.DataFrame()
    
    for n in dfs.keys():
        print('combine_results_by_depth: n=', n)
        print(dfs.keys())
        p = int(n[114:115])  # pid
        print('pid', p)
        l = n[106:107]  # task
        print('l', l)
        df = dfs.get(n)
        dft = df.T.add_prefix(f'{l}_')
        if frame.empty:
            frame = dft
        else:
            # print(dft)
            # print('Before joining')
            # print(frame)
            frame = frame.join(dft)
    return frame


def save_depth_file(df, depth):
    path = f'/Users/agnese/Dropbox/PyCharmProjects/gym_tower_of_london/training/model_simulations/experimental_results/by_depth/depth={depth}.csv'
    print('Saving file')
    df.to_csv(path)


def save_results_by_depth():
    # Read saved stats
    directory = f'/Users/agnese/Dropbox/PyCharmProjects' \
                f'/gym_tower_of_london/training/model_simulations' \
                f'/experimental_results/'
    files = [join(directory, f) for f in listdir(directory) if isfile(join(
        directory, f)) and not f.startswith('.DS_Store')]
    
    depth_0, depth_1, depth_2, depth_3, depth_4 = sort_files_by_depth(files)
    dfs_0, dfs_1, dfs_2, dfs_3, dfs_4 = get_file_dictionaries(depth_0,
                                                              depth_1,
                                                              depth_2,
                                                              depth_3,
                                                              depth_4)
    t0 = combine_results_by_depth(dfs_0)
    t1 = combine_results_by_depth(dfs_1)
    t2 = combine_results_by_depth(dfs_2)
    t3 = combine_results_by_depth(dfs_3)
    t4 = combine_results_by_depth(dfs_4)
    
    #
    save_depth_file(t0, 0)
    save_depth_file(t1, 1)
    save_depth_file(t2, 2)
    save_depth_file(t3, 3)
    save_depth_file(t4, 4)


def normalize_data():
    for depth in range(5):
        path = f'/Users/agnese/Dropbox/PyCharmProjects/gym_tower_of_london/training/model_simulations/experimental_results/by_depth/'
        name = join(path, f'depth={depth}.csv')
        df = pd.read_csv(name, index_col=False)
        for col in df.columns:
            if col == 'Unnamed: 0':
                continue
            start = col[0]
            min = get_min.get(start)
            df[col] = [v / min for v in df[col]]
        
        file_name = join(path, f'depth={depth}_normalized.csv')
        df.to_csv(file_name)


if __name__ == '__main__':
    save_last_move_statistics()
    save_results_by_depth()
    normalize_data()
