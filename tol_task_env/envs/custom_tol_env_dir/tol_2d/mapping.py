from envs.custom_tol_env_dir.tol_2d.state import TolState

state_to_int = {
    TolState(1, 1): 11,
    TolState(1, 2): 12,
    TolState(1, 3): 13,
    TolState(1, 4): 14,
    TolState(1, 5): 15,
    TolState(1, 6): 16,

    TolState(2, 1): 21,
    TolState(2, 2): 22,
    TolState(2, 3): 23,
    TolState(2, 4): 24,
    TolState(2, 5): 25,
    TolState(2, 6): 26,

    TolState(3, 1): 31,
    TolState(3, 2): 32,
    TolState(3, 3): 33,
    TolState(3, 4): 34,
    TolState(3, 5): 35,
    TolState(3, 6): 36,

    TolState(4, 1): 41,
    TolState(4, 2): 42,
    TolState(4, 3): 43,
    TolState(4, 4): 44,
    TolState(4, 5): 45,
    TolState(4, 6): 46,

    TolState(5, 1): 51,
    TolState(5, 2): 52,
    TolState(5, 3): 53,
    TolState(5, 4): 54,
    TolState(5, 5): 55,
    TolState(5, 6): 56,

    TolState(6, 1): 61,
    TolState(6, 2): 62,
    TolState(6, 3): 63,
    TolState(6, 4): 64,
    TolState(6, 5): 65,
    TolState(6, 6): 66,
}

int_to_state = {
    11: TolState(1, 1),
    12: TolState(1, 2),
    13: TolState(1, 3),
    14: TolState(1, 4),
    15: TolState(1, 5),
    16: TolState(1, 6),

    21: TolState(2, 1),
    22: TolState(2, 2),
    23: TolState(2, 3),
    24: TolState(2, 4),
    25: TolState(2, 5),
    26: TolState(2, 6),

    31: TolState(3, 1),
    32: TolState(3, 2),
    33: TolState(3, 3),
    34: TolState(3, 4),
    35: TolState(3, 5),
    36: TolState(3, 6),

    41: TolState(4, 1),
    42: TolState(4, 2),
    43: TolState(4, 3),
    44: TolState(4, 4),
    45: TolState(4, 5),
    46: TolState(4, 6),

    51: TolState(5, 1),
    52: TolState(5, 2),
    53: TolState(5, 3),
    54: TolState(5, 4),
    55: TolState(5, 5),
    56: TolState(5, 6),

    61: TolState(6, 1),
    62: TolState(6, 2),
    63: TolState(6, 3),
    64: TolState(6, 4),
    65: TolState(6, 5),
    66: TolState(6, 6),
}
