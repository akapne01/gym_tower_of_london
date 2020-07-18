from training.utils.tree_methods import get_min_no_moves

"""
Experimental tasks that correspond to tasks in Experiment 2 (Cooper & Marsh, 2016)
"""
tasks = {
    'A': {
        'start': 53,
        'goal': 14,
        'min_moves': get_min_no_moves(53, 14)
    },

    'B': {
        'start': 11,
        'goal': 52,
        'min_moves': get_min_no_moves(11, 52)
    },
    'C': {
        'start': 34,
        'goal': 56,
        'min_moves': get_min_no_moves(34, 56)
    },
    'D': {
        'start': 46,
        'goal': 16,
        'min_moves': get_min_no_moves(46, 16)
    },
    'E': {
        'start': 33,
        'goal': 52,
        'min_moves': get_min_no_moves(33, 52)
    },
    'F': {
        'start': 13,
        'goal': 32,
        'min_moves': get_min_no_moves(13, 32)
    }
}
if __name__ == '__main__':
    print('A', tasks.get('A'))
    print('B', tasks.get('B'))
    print('C', tasks.get('C'))
    print('D', tasks.get('D'))
    print('E', tasks.get('E'))
    print('F', tasks.get('F'))
