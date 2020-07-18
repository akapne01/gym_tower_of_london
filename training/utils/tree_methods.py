import random
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import pydot
from anytree import Node
from anytree.exporter import DotExporter

from training.utils.mapping import int_to_state
from training.utils.planning_helper import get_possible_actions, \
    state_ball_mapper

depth = 3


def no_in_positions(state, goal):
    # get current ball positions
    red, green, blue = state_ball_mapper.get(state)

    # get goal ball positions
    red_goal, green_goal, blue_goal = state_ball_mapper.get(goal)

    # Init counter
    balls_in_goal_place = 0

    if red == red_goal:
        balls_in_goal_place += 1

    if green == green_goal:
        balls_in_goal_place += 1

    if blue == blue_goal:
        balls_in_goal_place += 1
    return balls_in_goal_place


def find_max(tree, state):
    """
    Uses depth first search to find the max
    value in the tree that is underneath state
    :param tree: tree
    :param state: int
    :return: maximum future reward
    """
    max = 0
    for a, b in nx.dfs_edges(tree, state):
        value = tree[a][b]['weight']
        if value > max:
            max = value
    print(
        f'find_max: After travesing the tree to find max for state {state}, the max value is',
        max)
    return max


def plan_for_best_actions(state, tree) -> List:
    """
    Traverses the tree and returns list representing
    the next possible actions that yield the highest
    reward.
    :param tree: look-ahead tree of certain depth
    constructed with rewards added for each state
    :param state: int
    :return: List
    """
    max_planned = 0
    best_children = []
    print(state)
    try:
        actions = tree.neighbors(state)
        for node in actions:
            value = find_max(tree, node)
            if value > max_planned and node != state:
                max_planned = value
                best_children.append(node)
        print('## plan_for_best_actions returns:', best_children)
        return best_children
    except AttributeError:
        return []


#
# def look_ahead(state, depth, start_state, goal_state, moves_made, tree):
#     """
#     Returns List of actions with the best rewards
#     :param state:
#     :param depth:
#     :param start_state:
#     :param goal_state:
#     :param moves_made:
#     :return:
#     """
#     # g = search_tree(state, depth, start_state, goal_state, moves_made)
#     return plan_for_best_actions(tree, state)
#
# def look_ahead(state, tree):
#     """
#     Returns List of actions with the best rewards
#     :param state:
#     :param depth:
#     :param start_state:
#     :param goal_state:
#     :param moves_made:
#     :return:
#     """
#     # g = search_tree(state, depth, start_state, goal_state, moves_made)
#     return plan_for_best_actions(tree, state)


def add_level_end_reward(nodes, g, start_position, goal_state, moves_made,
                         states_in_tree) -> None:
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
            add_level_end_reward(n, g, start_position, goal_state,
                                 moves_made + 1, states_in_tree)
    else:
        children = get_possible_actions(nodes)
        g.add_nodes_from(children)
        for a in children:
            if a in states_in_tree:
                # print(f'Node {a} is already int the tree, continuing')
                continue
            min_moves = get_min_no_moves(start_position, goal_state)
            # print(f'# add_level_end_reward : adding reward for node {nodes}')
            g.add_edge(nodes, a,
                       weight=calculate_end_reward(nodes, goal_state,
                                                   min_moves, moves_made))

        states_in_tree.update(children)


def add_one_level_step_reward(nodes, g, start_position, goal, moves_so_far,
                              states_in_tree) -> None:
    """
    Adds level with rewards on every step
    :param nodes: can be an int or a list
    :param g: graph
    :param start_position: start position of the problem
    :param goal: goal state for the problem
    :param moves_so_far: how many moves are simulated
    :param states_in_tree: which states have already been added to the tree
    """

    # nodes can be a state or a list of nodes
    if isinstance(nodes, list):
        for n in nodes:
            add_one_level_step_reward(n, g, start_position, goal,
                                      moves_so_far + 1, states_in_tree)
    else:
        children = get_possible_actions(nodes)
        g.add_nodes_from(children)

        """
        Loop through children and copy the q_values, visits and 
        """
        n_goal = no_in_positions(nodes, goal)
        for a in children:
            if a in states_in_tree:
                continue
            g.add_edge(nodes, a,
                       weight=calculate_step_reward(action=a,
                                                    goal_state=goal,
                                                    start_position=start_position,
                                                    moves_made=moves_so_far,
                                                    n_goal_pos=n_goal))
        # states_in_tree.update(children)


def search_tree(state, depth, start_state, goal_state, moves_made):
    """
    Searach tree that uses rewards applied at every step

    :param state:
    :param depth:
    :param start_state:
    :param goal_state:
    :param moves_made:
    :param n_goal: number of balls in the goal position
    :return:
    """
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
    n_goal = no_in_positions(state, goal_state)
    add_one_level_step_reward(states, g, start_state, goal_state, moves_made,
                              states_in_tree, n_goal)
    moves_made += 1
    """
    Add higher level lookahead
    """
    states = get_possible_actions(state)

    for i in range(depth - 1):
        next_states = []
        for s in states:  # add one level for each of the child nodes
            n_goal = no_in_positions(s, goal_state)
            add_one_level_step_reward(s, g, start_state, goal_state,
                                      moves_made, states_in_tree, n_goal)
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


def search_graph(state, depth, start_state, goal_state, moves_made):
    """
    Searach tree that uses rewards applied at the end of the game

    :param state:
    :param depth:
    :param start_state:
    :param goal_state:
    :param moves_made:
    :param n_goal: number of balls in the goal position
    :return:
    """
    # create graph
    # g = nx.DiGraph()
    g = nx.Graph()

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
    # n_goal = no_in_positions(state, goal_state)
    add_one_level_step_reward(states, g, start_state, goal_state, moves_made,
                              states_in_tree)
    moves_made += 1
    """
    Add higher level lookahead
    """
    states = get_possible_actions(state)

    for i in range(depth - 1):
        next_states = []
        for s in states:  # add one level for each of the child nodes
            add_one_level_step_reward(s, g, start_state, goal_state,
                                      moves_made, states_in_tree)
            next_states.extend(get_possible_actions(s))
        moves_made += 1
        states = next_states
    return g


def search_tree_end_rewards(state, depth, start_state, goal_state, moves_made):
    """
    Searach tree that uses rewards applied at the end of the game

    :param state:
    :param depth:
    :param start_state:
    :param goal_state:
    :param moves_made:
    :param n_goal: number of balls in the goal position
    :return:
    """
    # create graph
    g = nx.DiGraph()

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
    # n_goal = no_in_positions(state, goal_state)
    add_level_end_reward(states, g, start_state, goal_state, moves_made,
                         states_in_tree)
    moves_made += 1
    """
    Add higher level lookahead
    """
    states = get_possible_actions(state)

    for i in range(depth - 1):
        next_states = []
        for s in states:  # add one level for each of the child nodes
            add_level_end_reward(s, g, start_state, goal_state, moves_made,
                                 states_in_tree)
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
    G = nx.DiGraph()
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
    pos = nx.fruchterman_reingold_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'),
                           node_size=500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='b', arrows=True)
    plt.show()


def calculate_end_reward(action: int, goal: int, min_moves: int,
                         move_count: int) -> float:
    # print(f'# calculate_end_reward: action={action}, goal={goal}, min_moves={min_moves}, move_count={move_count}')
    actions_to_reward = get_possible_actions(goal)
    # print(f'# calculate_end_reward: actions to reward: {actions_to_reward}')
    if action in actions_to_reward:
        # if action == goal:
        if move_count == min_moves:
            # print('# calculate_end_reward : returning 100')
            return 100.0
        # print('# calculate_end_reward : returning 1')
        return 1.0
    # print('# calculate_end_reward : returning 0')
    return 0.0


def calculate_step_reward(action: int, goal_state: int,
                          start_position: int, moves_made: int,
                          n_goal_pos: int) -> float:
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
    # Calculate Reward
    before = n_goal_pos
    balls_in_goal_place = no_in_positions(action, goal_state)
    min_moves = get_min_no_moves(start_position, goal_state)

    if action == goal_state:
        if moves_made == min_moves:
            return 100 / moves_made
        return 1 / moves_made
    if balls_in_goal_place > before:
        return 0.25 / moves_made
    elif balls_in_goal_place == before:
        return 0
    else:
        return (-1) * 0.25 / moves_made


def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5,
                   pos=None, parent=None):
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


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0,
                  xcenter=0.5):
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
        raise TypeError(
            'cannot use hierarchy_pos on a graph that is not a tree')

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
                edgeattrfunc=lambda parent,
                                    child: "style=bold,label=%10.2f" % (
                        child.reward or 0)
                ).to_dotfile(f"images/dot/reward_tree_{s}.dot")
    (graph,) = pydot.graph_from_dot_file(f"images/dot/reward_tree_{s}.dot")
    graph.write_png(f"images/png/reward_tree_{s}.png")


# def estimate_v_planning(depth, width, gamma, model, state):
def estimate_v_planning(depth, model, state, Q):
    print(f'# estimate_v_planning : depth={depth}, state = {state}')
    actions = get_possible_actions(state)
    q_values = []

    for a in actions:
        if depth == 0:
            q_values.append(Q.loc[a, state])
        else:
            q_values.append(estimate_v_planning(depth - 1, model, a, Q))
    max_value = max(q_values)
    print(f'# estimate_v_planning: returning max_value={max_value}')
    return max(q_values)


def look_ahead_plan_future_returns(depth, state, start, goal, moves_made):
    # Estimate Rewards
    """
    Each call to this function simulates imagined move made and observing
    the reward that was returned by that action.
    :param depth:
    :param state:
    :param start:
    :param goal:
    :param moves_made:
    :return:
    """
    print(f'# look_ahead_plan_future_returns : depth={depth}, state = {state}')
    actions = get_possible_actions(state)
    action_values = []
    for a in actions:
        if depth == 0:
            n_goal = no_in_positions(state, goal)
            action_values.append(
                calculate_step_reward(a, goal, start, moves_made, n_goal))
        else:
            action_values.append(
                look_ahead_plan_future_returns(depth - 1, a, start, goal,
                                               moves_made + 1))
    max_value = max(action_values)
    print(f'# estimate_v_planning: returning max_value={max_value}')
    return max(action_values)


# def estimate_q_planning(depth, width, gamma, model, state):
def estimate_q_planning(depth, model, state, Q) -> List:
    print(f'# estimate_q_planning: state={state}, depth={depth}')
    actions = get_possible_actions(state)
    q_values = []
    for a in actions:
        q_values.append(estimate_v_planning(depth - 1, model, a, Q))
    print(f'# estimate_q_planning:returning q_values = {q_values}')
    return q_values


def estimate_action_rewards_using_planning(depth, state, start, goal, moves_made) -> List:
    # Plan rewards
    """
    For all the actions that  are possible to be taken from the
    state gets the max look-ahead values for specified look-ahead
    depth
    :param depth:
    :param state:
    :param start:
    :param goal:
    :param moves_made:
    :return: List of next action values
    """
    print(f'# estimate_action_rewards_using_planning: state={state}, depth={depth}')
    actions = get_possible_actions(state)
    action_values = []

    for a in actions:
        # test this
        if depth == 0:
            n_goal = no_in_positions(state, goal)
            action_values.append(
                calculate_step_reward(a, goal, start, moves_made, n_goal))
        else:
            action_values.append(
                look_ahead_plan_future_returns(depth - 1, a, start, goal,
                                               moves_made))
    print(f'# estimate_q_planning:returning action_values = {action_values}')
    return action_values


# def planning_algorithm(epsilon, gamma, max_reward, model, state, depth):
def planning_algorithm(model, state, depth, Q):
    print(f'# planning_algorithm: state={state}, depth={depth}')
    actions = get_possible_actions(state)
    q_values = estimate_q_planning(depth, model, state, Q)
    max_value = max(q_values)
    for a, v in zip(actions, q_values):
        if v == max_value:
            print(f'# planning_algorithm: returning action a = {a}')
            return a
    print(f'# planning_algorithm: returning action a = NONE')
    return None


def planning_algorithm_new(state, depth, start, goal, moves_made):
    """
    Finds the best action that can be taken in the state. This action is
    found using look-ahead tree. Algorithm finds the maximum reward that
    can be obtained after sepcified lookahead.
    :param state:
    :param depth: represent how many moves ahead algorithm looks for the
    rewards.
    :param start:
    :param goal:
    :param moves_made:
    :return: Returns the best of immediate actions with look-ahead
    depth
    """
    # Selects the best action
    print(f'# planning_algorithm: state={state}, depth={depth}')

    actions = get_possible_actions(state)
    action_values = estimate_action_rewards_using_planning(depth, state, start, goal, moves_made)
    max_value = max(action_values)
    for a, v in zip(actions, action_values):
        if v == max_value:
            print(f'# planning_algorithm: returning action a = {a}')
            return a
    print(f'# planning_algorithm: returning action a = NONE')
    return None


def planning_algorithm_2(state, depth, start, goal, moves_made):
    # Looks for rewards
    print(f'# planning_algorithm: state={state}, depth={depth}')
    actions = get_possible_actions(state)
    q_values = estimate_action_rewards_using_planning(depth, state, start, goal, moves_made)
    max_value = max(q_values)
    for a, v in zip(actions, q_values):
        if v == max_value:
            print(f'# planning_algorithm: returning action a = {a}')
            return a
    print(f'# planning_algorithm: returning action a = NONE')
    return None


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
    """
    Populates png files with search trees for all possible ToL problems
    """
    # for i in int_to_state.keys():
    #     for j in int_to_state.keys():
    #         if i != j:
    i = 33
    j = 54

    start_state = i
    goal_state = j
    moves_made = 0
    # depth = get_min_no_moves(i, j)
    depth = 3
    print('Depth: ', depth)
    G = problem_space_graph()
    state = i
    # tree = nx.dfs_tree(G, 33)
    # n_goal = no_in_positions(state, goal_state)
    # tree = search_tree(state, depth, start_state, goal_state, moves_made)
    # tree = search_tree_end_rewards(state, 4, start_state, goal_state, moves_made)
    tree = search_graph(state, depth, start_state, goal_state, moves_made)
    print('~~ Get values from tree ~~')
    for n in tree.neighbors(33):
        print(n)
    # print(tree)

    # Finding which of the actions have a better reward:
    print('Find better action:')

    # pos = nx.spring_layout(G)
    # nx.draw(G, pos=pos, with_labels=True, node_size=600, node_color='lightgreen')

    # labels = nx.get_edge_attributes(G, 'weight')

    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='black')
    # nx.draw_networkx_edges(G, pos, edge_color='green', arrows=True)

    # To draw a tree
    # pos = hierarchy_pos(tree, state)
    pos = nx.spring_layout(tree, state)
    nx.draw(tree, pos=pos, with_labels=True, node_size=600,
            node_color='lightgreen')
    #
    labels = nx.get_edge_attributes(tree, 'weight')
    #
    nx.draw_networkx_edge_labels(tree, pos, edge_labels=labels,
                                 font_color='black')
    nx.draw_networkx_edges(tree, pos, edge_color='green', arrows=True)
    # plt.savefig(f'tree_{i}_{j}_{depth}.png')
    #
    plt.show()

    # draw_problem_space()
