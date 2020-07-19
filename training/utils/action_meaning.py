from training.utils.state_mapping import state_ball_mapper


def get_action_meaning(s1: int, s2: int) -> str:
    """
    Returns the change between 2 states as an action meaning.
    Example: [G C2 -> G R2]
    G represent colour, possible colours are R - red,
    G - green, B - blue.
    C2 represents rod and position of rod. Left rod holds
    only 1 position, Center rod holds 2 positions, Right
    rod can hold 3 positions.
    """
    r1, g1, b1 = state_ball_mapper.get(s1)
    r2, g2, b2 = state_ball_mapper.get(s2)
    red_moved = r2 - r1 != 0
    green_moved = g2 - g1 != 0
    blue_moved = b2 - b1 != 0
    colour = ''
    rod_from = ''
    rod_to = ''
    
    if red_moved:
        colour = 'R'
        arr1, arr2 = r1 % 10, r2 % 10
        rod_from = get_rod_letter(r1)
        rod_to = get_rod_letter(r2)
    
    if green_moved:
        colour = 'G'
        arr1, arr2 = g1 % 10, g2 % 10
        rod_from = get_rod_letter(g1)
        rod_to = get_rod_letter(g2)
    
    if blue_moved:
        colour = 'B'
        arr1, arr2 = b1 % 10, b2 % 10
        rod_from = get_rod_letter(b1)
        rod_to = get_rod_letter(b2)
    
    action = f'[{colour} {rod_from}{arr1} -> {colour} {rod_to}{arr2}]'
    return action


def get_rod_letter(position: int):
    rod = position // 10
    if rod == 1:
        return 'L'
    if rod == 2:
        return 'C'
    if rod == 3:
        return 'R'
    return None


if __name__ == '__main__':
    print('### 11')
    print(get_action_meaning(11, 12))
    print(get_action_meaning(11, 13))
    print('### 12')
    print(get_action_meaning(12, 11))
    print(get_action_meaning(12, 13))
    print(get_action_meaning(12, 25))
    print('### 13')
    print(get_action_meaning(13, 11))
    print(get_action_meaning(13, 12))
    print(get_action_meaning(13, 14))
    print(get_action_meaning(13, 15))
    print('### 14')
    print(get_action_meaning(14, 13))
    print(get_action_meaning(14, 15))
    print(get_action_meaning(14, 66))
    print('### 15')
    print(get_action_meaning(15, 13))
    print(get_action_meaning(15, 14))
    print(get_action_meaning(15, 16))
    print(get_action_meaning(15, 22))
    print('### 16')
    print(get_action_meaning(16, 15))
    print(get_action_meaning(16, 64))
