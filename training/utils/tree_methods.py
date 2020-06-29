import random

import matplotlib.pyplot as plt
import networkx as nx
import pydot
from anytree import Node
from anytree.exporter import DotExporter

from training.utils.mapping import int_to_state
from training.utils.planning_helper import get_possible_actions, state_ball_mapper

depth = 3


def add_one_level(nodes, g, start_position, goal_state, moves_made, states_in_tree) -> None:
    """
    :param nodes: can be an int or a list
    :param g: graph
    :param start_position: start position of the problem
    :param goal_state: goal state for the problem
    :param moves_made: how many moves are simulated
    :param states_in_tree: which states have already been added to the tree
    """
    # nodes can be a state or a list of nodes
    if isinstance(nodes, list):
        for n in nodes:
            add_one_level(n, g, start_position, goal_state, moves_made + 1, states_in_tree)
    else:
        children = get_possible_actions(nodes)
        g.add_nodes_from(children)
        for a in children:
            if a in states_in_tree:
                continue
            g.add_edge(nodes, a,
                       weight=calculate_weighted_reward(action=a, goal_state=goal_state, start_position=start_position,
                                                        moves_made=moves_made))
            print(
                'calculate_weighted_reward(action=a, goal_state=goal_state, start_position=start_position,moves_made=moves_made)')
            print('action', a)
            print('goal', goal_state)
            print('start', start_position)
            print('moves made', moves_made)
            print()

        states_in_tree.update(children)


def search_tree(state, depth, start_state, goal_state, moves_made):
    # create graph
    g = nx.DiGraph()
    # g = nx.MultiDiGraph()

    # States that are added to the tree are saved in this set
    states_in_tree = set()
    states_in_tree.add(state)
    """
    Add current node
    """
    g.add_node(state)
    moves_made += 1
    """
    Add 1 level lookahead
    """
    states = state
    add_one_level(states, g, start_state, goal_state, moves_made, states_in_tree)
    moves_made += 1
    """
    Add higher level lookahead
    """
    states = get_possible_actions(state)

    for i in range(depth - 1):
        next_states = []
        for s in states:
            add_one_level(s, g, start_state, goal_state, moves_made, states_in_tree)
            next_states.extend(get_possible_actions(s))
        moves_made += 1
        states = next_states
    """
    Add 3rd level lookahead
    """
    # next_states = []
    # for s in states:
    #     print('3rd level lookahead')
    #     print('s:', s)
    #     add_one_level(s, g, start_state, goal_state, moves_made, states_in_tree)
    #     next_states.extend(get_possible_actions(s))
    # print('After Level 3:')

    """
    Plot graph 
    """
    #
    # pos = nx.spring_layout(g)
    # # pos = nx.planar_layout(g)
    # nx.draw_networkx_nodes(g, pos,
    #                        cmap=plt.get_cmap('jet'),
    #                        node_size=500)
    # nx.draw_networkx_labels(g, pos)
    # labels = nx.get_edge_attributes(g, 'weight')
    # print('Labels are', labels)
    #
    # nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
    # nx.draw_networkx_edges(g, pos, edge_color='b', arrows=True)
    # plt.show()
    #

    # Tree
    # T = nx.full_rary_tree(create_using=g)
    #
    # pos = graphviz_layout(T, prog="dot")
    # nx.draw(T, pos)
    # plt.show()
    return g


#
#
# def add_one_level_reward_lookahead(node: Node, goal_state: int, start_position: int, moves_made: int) -> Node:
#     """
#     Adds children to the node tree to represent a one level lookahead
#     :param moves_made:
#     :param node:
#     :param goal_state:
#     :param start_position:
#     :return:
#     """
#     possible_actions = get_possible_actions(node.name)
#     child_nodes = [
#         RewardNode(a, reward=calculate_weighted_reward(action=a, goal_state=goal_state, start_position=start_position,
#                                                        moves_made=moves_made))
#         for a in
#         possible_actions]
#     node.children = child_nodes
#     return node


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


def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    """
    :param G: the graph (must be a tree)
    :param root: the root node of current branch
    :param width: horizontal space allocated for this branch
    :param vert_gap: gap between levels of hierarchy
    :param vert_loc: vertical location of root
    :param xcenter: horizontal location of root
    :param pos: a dict saying where all nodes go if they have been assigned
    :param parent: parent of this branch. - only affects it if non-directed
    """
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                 vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                 pos=pos, parent=root)
    return pos


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    :param G: the graph (must be a tree)
    :param root: the root node of current branch
     - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.
    :param width: horizontal space allocated for this branch - avoids overlap with other branches
    :param vert_gap: gap between levels of hierarchy
    :param vert_loc: vertical location of root
    :param xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))
        else:
            root = random.choice(list(G.nodes))

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def save_tree(s, tree: Node) -> None:
    """
    Saves tree as a *.dot file then reads the
    dot file that is then converted to a png
    file
    :param tree: Node to save
    """
    DotExporter(tree,
                nodenamefunc=lambda node: node.name,
                edgeattrfunc=lambda parent, child: "style=bold,label=%10.2f" % (child.reward or 0)
                ).to_dotfile(f"images/dot/reward_tree_{s}.dot")
    (graph,) = pydot.graph_from_dot_file(f"images/dot/reward_tree_{s}.dot")
    graph.write_png(f"images/png/reward_tree_{s}.png")


if __name__ == '__main__':
    # print('Min 16, 25: ', get_shortest_path(16, 25))
    # print('0 = Min 16, 25: ', get_min_no_moves(16, 25))
    # print('A =  53 - 14 : ', get_min_no_moves(53, 14))
    # print('B =  11 - 52 : ', get_min_no_moves(11, 52))
    # print('C =  34 - 56 : ', get_min_no_moves(34, 56))
    # print('D =  46 - 16 : ', get_min_no_moves(46, 16))
    # print('E =  33 - 52 : ', get_min_no_moves(33, 52))
    # print('F =  13 - 32 : ', get_min_no_moves(13, 32))
    # draw_problem_space()

    start_state = 16
    goal_state = 25
    moves_made = 0
    depth = 3
    g = search_tree(16, depth, start_state, goal_state, moves_made)

    pos = hierarchy_pos(g, 16)
    nx.draw(g, pos=pos, with_labels=True)

    labels = nx.get_edge_attributes(g, 'weight')

    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
    nx.draw_networkx_edges(g, pos, edge_color='b', arrows=True)
    # plt.savefig('search_tree.png')
    plt.show()
