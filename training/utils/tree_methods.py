import random
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pydot
from anytree import Node
from anytree.exporter import DotExporter

from training.utils.planning_helper import get_possible_actions
from training.utils.state_mapping import int_to_state, state_ball_mapper


def no_in_positions(state: int, goal: int) -> int:
    """
    Calculates how many balls in the current state are
    in goal positions
    :return: number of balls in a goal positions
    """
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


def calculate_step_reward(action: int,
                          goal_state: int,
                          start_position: int,
                          moves_made: int,
                          state: int) -> float:
    """
    Positive reward is given when a ball is put in the goal
    position, that was not in a goal position before.
    Negative reward is given when a ball that was previously
    in a goal position is moved away from the goal position.
    Rewards are move dependent, as the main objective is to
    solve the task in minimum moves.
    If game completed in minimum rewards then the reward
    returned is 100 divided by number of minimum moves.

    To determine positive or negative reward, calculate how
    many balls were in goal position before the action is
    taken, and after the action is taken. If new balls are
    put in goal position then positive reward, if ball taken
    away, negative reward is obtained.
    """
    before = no_in_positions(state, goal_state)
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


def get_q_value(state, action, q_values):
    return q_values.loc[action, state]


def add_one_level_q_values(nodes, tree, start, goal, q_values,
                           states_in_tree) -> None:
    """
    Adds level with rewards on every step
    :param nodes: can be an int or a list
    :param tree: graph
    :param start_position: start position of the problem
    :param goal: goal state for the problem
    :param moves_so_far: how many moves are simulated
    :param states_in_tree: which states have already been added to the tree
    """
    
    # nodes can be a state or a list of nodes
    if isinstance(nodes, list):
        for n in nodes:
            add_one_level_step_reward(n, tree, start, goal, q_values,
                                      states_in_tree)
    else:
        children = get_possible_actions(nodes)
        # g.add_nodes_from(children)
        
        """
        Loop through children and copy the q_values, visits and
        """
        for a in children:
            if a in states_in_tree:
                continue
            tree.add_edge(nodes, a,
                          weight=get_q_value(state=nodes, action=a, q_values=q_values))
            states_in_tree.add(a)


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
        # g.add_nodes_from(children)
        
        """
        Loop through children and copy the q_values, visits and 
        """
        for a in children:
            if a in states_in_tree:
                continue
            g.add_edge(nodes, a,
                       weight=calculate_step_reward(action=a,
                                                    goal_state=goal,
                                                    start_position=start_position,
                                                    moves_made=moves_so_far,
                                                    state=nodes))
            states_in_tree.add(a)


def build_q_value_tree(state, depth, start_state, goal_state, q_values):
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
    tree = nx.DiGraph()
    
    # States that are added to the tree are saved in this set
    states_in_tree = set()
    states_in_tree.add(state)
    """
    Add current node
    """
    tree.add_node(state)
    
    """
    Add 1 level lookahead
    """
    states = state
    add_one_level_q_values(states, tree, start_state, goal_state, q_values,
                           states_in_tree)
    
    # add_one_level_step_reward(states, tree, start_state, goal_state, moves_made,
    #                           states_in_tree)
    
    """
    Add higher level lookahead
    """
    states = get_possible_actions(state)
    
    for i in range(depth - 1):
        next_states = []
        for s in states:  # add one level for each of the child nodes
            add_one_level_q_values(states, tree, start_state, goal_state,
                                   q_values,
                                   states_in_tree)
            next_states.extend(get_possible_actions(s))
        states = next_states
    return tree


def build_search_tree(state, depth, start_state, goal_state, moves_made):
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


def look_for_rewards_in_tree(state: int,
                             depth: int,
                             start: int,
                             goal: int,
                             moves_made: int) -> List:
    action_values = []
    actions = get_possible_actions(state)
    tree = build_search_tree(state=state, depth=depth, start_state=start,
                             goal_state=goal,
                             moves_made=moves_made)
    for a in actions:
        r = calculate_step_reward(action=a,
                                  goal_state=goal,
                                  start_position=start,
                                  moves_made=moves_made + 1,
                                  state=state)
        max_value = max(r, find_max(tree, a))
        action_values.append(max_value)
        print(f'# a = {a} & max = {max_value}')
    return action_values


def look_for_q_values_in_tree(state: int,
                              depth: int,
                              start: int,
                              goal: int,
                              moves_made,
                              q_values: pd.DataFrame) -> List:
    action_values = []
    actions = get_possible_actions(state)
    tree = build_q_value_tree(state, depth, start, goal, q_values)
    for a in actions:
        r = calculate_step_reward(action=a,
                                  goal_state=goal,
                                  start_position=start,
                                  moves_made=moves_made + 1,
                                  state=state)
        max_value = max(r, find_max(tree, a))
        action_values.append(max_value)
        print(f'# a = {a} & max = {max_value}')
    return action_values


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
    actions = tree.neighbors(state)
    
    for node in actions:
        value = find_max(tree, node)
        if value > max_planned and node != state:
            max_planned = value
            best_children.append(node)
    print('## plan_for_best_actions returns:', best_children)
    
    return best_children


def look_ahead(state, depth, start_state, goal_state, moves_made):
    """
    Returns List of actions with the best rewards
    :param state:
    :param depth:
    :param start_state:
    :param goal_state:
    :param moves_made:
    :return:
    """
    tree = build_search_tree(state, depth, start_state, goal_state, moves_made)
    return plan_for_best_actions(tree, state)


if __name__ == '__main__':
    """
    Populates png files with search trees for all possible ToL problems
    """
    i = 23
    j = 62
    
    start_state = i
    goal_state = j
    moves_made = 0
    
    depth = 3
    print('Depth: ', depth)
    # G = problem_space_graph()
    state = i
    
    v = look_for_rewards_in_tree(state=23, depth=4, start=23, goal=62,
                                 moves_made=0)
    print(f'find_heuristic_action returned {v}')
    
    # def find_max_children(state, depth, start, goal, moves):
    #     actions = get_possible_actions(state)
    #     tree = build_search_tree(state=state, depth=depth, start_state=start,
    #                              goal_state=goal,
    #                              moves_made=moves)
    #
    
    # To draw a tree
    tree = build_search_tree(state=23, depth=5, start_state=23,
                             goal_state=62,
                             moves_made=0)
    # av = plan_for_best_actions(23, tree)
    # print(f'plan_for_best_actions = {av}')
    
    pos = hierarchy_pos(tree, state)
    # pos = nx.spring_layout(tree, state)
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
    
    draw_problem_space()
