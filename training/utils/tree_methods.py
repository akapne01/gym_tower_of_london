import matplotlib.pyplot as plt
import networkx as nx
from anytree import Node
from anytree import NodeMixin

from training.utils.mapping import int_to_state
from training.utils.planning_helper import get_possible_actions, state_ball_mapper

depth = 3


def construct_planning_tree(state: int) -> Node:
    """
    Supports depth 1, 2, 3
    :param state:
    :return:
    """
    root = Node(state)
    possible_actions = get_possible_actions(state)
    child_nodes = [Node(a) for a in possible_actions]
    root.children = child_nodes
    if depth == 1:
        return root

    for r in root.children:
        possible_actions = get_possible_actions(r.name)
        child_nodes = [Node(a) for a in possible_actions]
        r.children = child_nodes

        if depth == 3:
            for node in child_nodes:
                add_one_level_lookahead(node)
    return root


def add_one_level_lookahead(node: Node) -> Node:
    possible_actions = get_possible_actions(node.name)
    child_nodes = [Node(a) for a in possible_actions]
    node.children = child_nodes
    return node


class RewardNode(NodeMixin):

    def __init__(self, name: int, reward=None, parent=None):
        """
        Node class that allows to add reward to the tree edges
        :param name: Represents state number
        :param reward: Reward to assign for the move
        :param parent: Parent Node
        """
        super(RewardNode, self).__init__()
        print(f'Creating node {name} with reward {reward}')
        self.name = name
        self.parent = parent
        self.reward = reward

    def _post_detach(self, parent):
        self.reward = None

    def __str__(self):
        return f'Action: {self.name}, Reward: {self.reward}'


def construct_reward_planning_tree(state: int, goal_state: int, start_position: int, moves_made: int):
    """
    Supports depth 1, 2, 3
    :param moves_made:
    :param start_position:
    :param goal_state:
    :param state:
    :return:
    """
    root = RewardNode(state)
    possible_actions = get_possible_actions(state)
    child_nodes = [
        RewardNode(a, reward=calculate_weighted_reward(action=a, goal_state=goal_state, start_position=start_position,
                                                       moves_made=moves_made))
        for a in
        possible_actions]
    root.children = child_nodes
    if depth == 1:
        return root

    for r in root.children:
        possible_actions = get_possible_actions(r.name)
        child_nodes = [RewardNode(a, reward=calculate_weighted_reward(action=a, goal_state=goal_state,
                                                                      start_position=start_position,
                                                                      moves_made=moves_made)) for a in
                       possible_actions]
        r.children = child_nodes

        if depth == 3:
            for node in child_nodes:
                add_one_level_reward_lookahead(node, goal_state, start_position, moves_made)
    return root


def add_one_level_reward_lookahead(node: Node, goal_state: int, start_position: int, moves_made: int) -> Node:
    """
    Adds children to the node tree to represent a one level lookahead
    :param moves_made:
    :param node:
    :param goal_state:
    :param start_position:
    :return:
    """
    possible_actions = get_possible_actions(node.name)
    child_nodes = [
        RewardNode(a, reward=calculate_weighted_reward(action=a, goal_state=goal_state, start_position=start_position,
                                                       moves_made=moves_made))
        for a in
        possible_actions]
    node.children = child_nodes
    return node


def get_shortest_path(start, goal):
    """
    Returns shortest path. For this problem space
    some of the problems have more than one problem
    space. This function returns the first occurrence
    it finds
    :param start: node from
    :param goal: node to
    :return: list of the path with the shortest walk from
    start node to goal node.
    """
    G = problem_space_graph()
    return nx.shortest_path(G, start, goal)


def get_min_no_moves(start, goal):
    """
    Takes as an input start and goal state numbers and
    return the minimum number of moves with what this
    problem can be solved.
    """
    G = problem_space_graph()
    return nx.shortest_path_length(G, start, goal)


def problem_space_graph() -> nx.Graph:
    """
    Creates a graph that represents the problem space
    of Tower of London Task
    :return: Graph
    """
    G = nx.Graph()
    li = []
    for state in int_to_state.keys():
        poss_next_states = get_possible_actions(state)
        l = [(state, s) for s in poss_next_states]
        li.extend(l)
    G.add_edges_from(li)
    return G


def draw_problem_space() -> None:
    """
    Plots the problem space of Tower of London task
    """
    G = problem_space_graph()
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'),
                           node_size=500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='b', arrows=True)
    plt.show()


def calculate_weighted_reward(action: int, goal_state: int, start_position: int, moves_made: int) -> float:
    """
    Only one action from all the possible actions that can be taken
    has a positive reward. All other actions have a negative reward.
    First, it calculates which action is leads closer to the the goal
    state and rewards that action.
    In order to calculate the action:
    1) Determine if the action leads directly to the goal state? If so,
    this action is rewarded.
    2) Else check if any action has the goal state colour permutation
    number if yes then finds which of these actions leads closer to the
    goal ball arrangement and selects that action.
    3) Else determines which action is the closest to goal states colour
    permutation number and selects it.
    :param moves_made:
    :param start_position:
    :param goal_state:
    :param state:
    :param action: action to use to calculate reward
    :return: +100 if action is rewarded, -100 is not
    """
    min_moves = get_min_no_moves(start_position, goal_state)
    red, green, blue = state_ball_mapper.get(action)
    red_goal, green_goal, blue_goal = state_ball_mapper.get(goal_state)
    balls_in_goal_place = 0
    if red == red_goal:
        balls_in_goal_place += 1

    if green == green_goal:
        balls_in_goal_place += 1

    if blue == blue_goal:
        balls_in_goal_place += 1

    if balls_in_goal_place == 3:
        if moves_made == min_moves:
            return 100
        return 1.0
    if balls_in_goal_place == 2:
        return 0.75

    if balls_in_goal_place == 1:
        return 0.5

    return 0


if __name__ == '__main__':
    print('Min 16, 25: ', get_shortest_path(16, 25))
    print('0 = Min 16, 25: ', get_min_no_moves(16, 25))
    print('A =  53 - 14 : ', get_min_no_moves(53, 14))
    print('B =  11 - 52 : ', get_min_no_moves(11, 52))
    print('C =  34 - 56 : ', get_min_no_moves(34, 56))
    print('D =  46 - 16 : ', get_min_no_moves(46, 16))
    print('E =  33 - 52 : ', get_min_no_moves(33, 52))
    print('F =  13 - 32 : ', get_min_no_moves(13, 32))
    draw_problem_space()
