"""
Script for saving summary files
"""

import os
from os import listdir
from os.path import isfile, join
import pandas as pd

letters = ['A', 'B', 'C', 'D', 'E']


def get_directory_dictionary():
    """
    Creates dictionary of directory names for each letter
    and each depth
    """
    directory = '/Users/agnese/Dropbox/PyCharmProjects/gym_tower_of_london/training/model_simulations/STATS/{}/rewards/depth_{}/'
    by_letter = {}
    for letter in letters:
        by_range = {}
        for depth in range(5):
            by_range[depth] = directory.format(letter, depth)
        by_letter[letter] = by_range
    return by_letter


def read_files_from_directory(directory):
    """
    Reads all files in directory and returns back
    list of the files that are in this directory
    """
    files = [join(directory, f) for f in listdir(directory) if
             isfile(join(directory, f)) and not f.startswith('summary')]
    return files


def save_files(files_dict, name):
    """
    Reads file names specified in directory
    and saves the summary file with the
    specified name
    """
    df = pd.DataFrame()
    for f in files_dict:
        frame = pd.read_csv(f, index_col=False)
        s = f.find('pid=')
        df[f[s: s + 6]] = frame.LAST_100_MOVES
    df.to_csv(name, index=False)


def save_summary():
    """
    Sums data for simulated participants in one file.
    There is a summary file saved for each problem and
    each depth.
    """
    directory = '/Users/agnese/Dropbox/PyCharmProjects/gym_tower_of_london/training/model_simulations/STATS/{}/rewards/depth_{}/'
    by_letter = get_directory_dictionary()
    for letter in letters:
        letter_dict = by_letter.get(letter)
        for depth in range(5):
            dir_depth = letter_dict.get(depth)
            dict_depth = read_files_from_directory(dir_depth)
            name = f'summary_{letter}_{depth}.csv'
            file_name = join(directory.format(letter, depth), name)
            save_files(dict_depth, file_name)


if __name__ == '__main__':
    save_summary()
