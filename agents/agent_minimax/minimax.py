import numpy as np
from typing import Optional
from enum import Enum
from agents.common import GameState, BoardPiece, PlayerAction, SavedState, check_end_state, apply_player_action


# Remark: general advice: functions should do one specific thing, and shouldn't be too long. Long docstrings are often a sign
#         that some refactoring should be done


""" Minimax algorithm.

This file contains all the necessary functions and values needed for the
execution of this minimax agent in the main.py file. For that,
the main function called "minimax_action" is used. It is important, that
is executed with the common.py contained in this project.

    Typical usage example:

        human_vs_agent(minimax_action)
    
"""


class BreakState(Enum):
    """ Class used to store a flag whether a Break/Continue must be performed.
   """
    BREAK_YES = 1
    BREAK_NO = -1


class Tree:
    """ Class used to create a branching tree.

    This creates the class so that a value can be given to each node,
    and thanks to the function "add_node", children can be appended
    to the node so more branches are attached to father nodes.

    Attributes:
        value : Heuristic value assigned to the node.
        child : Creation of a child to the node.

    """
    def __init__(self, value):
        """ Inits Tree assigning the value to the node."""
        self.value = value
        self.child = []

    def add_node(self, value):
        """ Appends the value to the child node."""
        self.child.append(Tree(value))

# Root of the tree (itself a node) and nodes should themselves be objects for bookkeeping.
# One simulation, create node, update statistics. Research maybe more simulation.


# Board values that will be used for the heuristic calculation of the nodes:
# It gives positions themselves a value, depending on the number of different
# possible outcomes can end up in a win when holding that position.

BOARD_VALUES = np.array([[3, 4, 5, 7, 5, 4, 3],
                         [4, 6, 8, 10, 8, 6, 4],
                         [5, 8, 11, 13, 11, 8, 5],
                         [5, 8, 11, 13, 11, 8, 5],
                         [4, 6, 8, 10, 8, 6, 4],
                         [3, 4, 5, 7, 5, 4, 3]])


# Remark: generally, I have a hard time understanding what you are doing. Your code and docstring is very long and
#         convoluted. This is a clear sign that refactoring is necessary.

# Remark: you should try to use more expressive names for variables and functions

def initialization_minimax_tree() -> Tree:
    """ Initializes the tree for the search of the best minimax outcome.
    Initializes child to be maximized with lowest possible value (- high value)
    and the ones which will be minimized with the biggest possible value (high value).

    Returns:
          An initialized tree of 4 layers of depth in a 7 columns board.
        {
          tree:
        }

    """
    val = 10000
    min_tree = Tree(-val)

    for i in range(0, 7):
        min_tree.add_node(-val)
        for j in range(0, 7):
            min_tree.child[i].add_node(val)
            for k in range(0, 7):
                min_tree.child[i].child[j].add_node(-val)
                for v in range(0, 7):
                    min_tree.child[i].child[j].child[k].add_node(+val)

    return min_tree


def assign_weight(board, pos_tree, player, board_values=BOARD_VALUES):
    """ Function assigning heuristic value to the addition of new piece.

    This function takes the board given and the column in which a new piece will
    be introduced by a specific player and return a heuristic value associated to
    that move.

    Args:
        board: C4 board of 6x7 to which a new action's heuristic will be tested.
        pos_tree: Column in which the player will deposit its piece.
        player: Player which is playing at that instant.
        board_values: Value pre-given to each position depending on the possible
                      wins one can perform in that position.

    Returns:
        game_state: GameState.IS_WIN if player won, GameState. Otherwise,
              GameState.STILL_PLAYING or GameState.IS_DRAW if board is full.
        val_board: Returns the heuristic value of the board_values, assigned to that board
                   position or returns 100 if the column is full and no piece can be
                   placed there.

    """
    old_board, position = apply_player_action(board, pos_tree, player, True, True)
    game_state = check_end_state(board, player)

    if position != 0:
        val_board = board_values[position]
    else:
        val_board = 100

    return game_state, val_board


def heuristic_evaluation(heuristic, board_val, i, idx, game, node_type=np.array([1])):
    """ Evaluation of the heuristic assigned to the node so that a break
        could be performed for.

    Intermediate evaluation of the heuristic value given by the assign_weight function.
    If the column was full, and thus a board value of 100 was given, the value of the node
    is assigned to +- high value (depending on the type of node), and a boolean variable
    is set to True, so that branch is no longer inspected (already full column), so that
    that column cannot be considered as a possible way of the development of the game.

    Args:
        heuristic: Represents a cumulative variable that takes the heuristic information
                   of previous movements up in the chain of depth layers so that the
                   heuristic conserves the importance of all the moves prior to this one.
        board_val: New board value associated to that new move in the respective column "i".
        i: Column used for the new piece placement.
        idx: Variable storing all the child nodes of this layer that have a plausible pathway
             to be inspected and followed (no pieces assigned to full columns or games that
             have already been won).
        game: GameState.IS_WIN if player won, GameState. Otherwise,
              GameState.STILL_PLAYING or GameState.IS_DRAW if board is full.
        node_type: 1 for minimizing node and -1 for maximizing node.

    Returns:
        break_y: True if a break is needed (full column or game is won), False otherwise.
        new_heuristic: Cumulative variable once it has been summed the value given by
                       the position of the new piece in the "i" column.

    """
    break_y = BreakState.BREAK_NO

    # Remark: Here specifically, you're abusing the evaluation of a node to signal that a column is full. Of course this
    #         works here, but it's mixing distinct features of the game and is not very clean.
    #         Another way of solving this is to only loop over valid moves, which needs to be checked beforehand.
    if board_val == 100:  # Full column.
        new_heuristic = node_type * 100000  # Child is not an option.
        break_y = BreakState.BREAK_YES
    else:
        '''Adding the board value to the cumulative variable if we are maximizing
        (the position gives more winning opportunities to the agent),
        or subtracting from the cumulative variable if minimizing (the opponent
        gets more chances of winning).'''

        new_heuristic = heuristic - node_type * board_val
        idx.append(i)

    if game == GameState.IS_WIN:
        new_heuristic = -node_type * 10000  # Choose this child.
        idx.append(i)
        break_y = BreakState.BREAK_YES

    return break_y, new_heuristic


def max_child(father, index):
    """ Function to this the child of a node maximizing it.

    Screening all of the child that were inspected (and there
    was no break when their heuristic assignation was performed) given by
    the variable index. If their value is the biggest one from them all, they
    will return that value, along with their position. If none of them was
    inspected, and thus all of them are equal to - high value, a random index will be
    returned.

    Args:
        father: Maximizing tree node.
        index: Indexes of which child has been assigned a heuristic value.

    Returns:
        idx: Index of the child with the biggest associated value.
        maxi: Value of that child.

    """
    maxi = -10000
    idx = np.random.randint(7)  # If variable 'index' is empty, give back a random integer.
    for k in index:
        new_val = father.child[k].value
        if new_val > maxi:
            maxi = new_val
            idx = k

    return idx, maxi


def min_child(father, index):
    """ Function to this the child of a node minimizing it.

        Screening all of the child that were inspected (and there
        was no break when their heuristic assignation was performed), given by
        the variable index. If their value is the smallest one from them all, they
        will return that value, along with their position. If none of them was
        inspected, and thus all of them are equal to a very high value, a random index will be
        returned.

        Args:
            father: Minimizing tree node.
            index: Indexes of which child has been assigned a heuristic value.

        Returns:
            idx: Index of the child with the smallest associated value.
            maxi: Value of that child.

        """
    mini = 10000
    idx = np.random.randint(7)
    for k in index:
        new_val = father.child[k].value
        if new_val < mini:
            mini = new_val
            idx = k

    return idx, mini


def minimax_action(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]):
    """Minimax agent getting a board and the corresponding player turn and returning
       the best non-full column for the player according to the algorithm.

    Enter the current state of the board, and performs a top-bottom search on different positions
    of the board, so that the most optimal according the heuristics used is found.

    Args:
        board: Current state of the board
        player: Whose turn is it.
        saved_state: Pre-computation work

    Returns:
        action: Best column to use.
        saved_state_out: Tree structure
    """
    global BOARD_VALUES
    # Remark: don't use global variables. Especially since you have a class here, just make it a class variable.

    tree = initialization_minimax_tree()  # Weights tree initialization.

    if player == BoardPiece(1):  # Finding out which player is who.
        other_player = BoardPiece(2)
    elif player == BoardPiece(2):
        other_player = BoardPiece(1)

    idx1 = []
    start = -1

    for i in range(0, 7):  # Player plays
        cumulative_values1 = 0 # Initialization of the cumulative variable.
        old_board = board.copy()

        # Optimal way to start: central column.
        if sum(sum(old_board[:, :]) == 0) == 7:
            start = 10   # Remark: another numerical flag, cf. the comment on board_val and also pos to flag a full column
            break

        # Remark: This next block would be much easier to read with better names
        game, board_val = assign_weight(old_board, i, player, BOARD_VALUES)
        break_y, cumulative_values1 = heuristic_evaluation(cumulative_values1, board_val, i, idx1, game, node_type=np.array([-1]))

        if break_y == BreakState.BREAK_YES and cumulative_values1 > 10000:  # Already a winning position, break the search.
            tree.child[i].value = cumulative_values1
            break
        elif break_y == BreakState.BREAK_YES and cumulative_values1 < 10000:  # Full column, do not go down its branches.
            tree.child[i].value = cumulative_values1
            continue

        idx2 = []

        for j in range(0, 7):  # other player plays
            old_board1 = old_board.copy()

            game, board_val = assign_weight(old_board1, j, other_player, BOARD_VALUES)
            break_y, cumulative_values2 = heuristic_evaluation(cumulative_values1, board_val, j, idx2, game)

            if break_y == BreakState.BREAK_YES:  # Either a full-column (worst value given) or a win (best one given).
                tree.child[i].child[j].value = cumulative_values2
                continue

            idx3 = []
            for k in range(0, 7):  # player plays
                old_board2 = old_board1.copy()

                game, board_val = assign_weight(old_board2, k, player, BOARD_VALUES)
                break_y, cumulative_values3 = heuristic_evaluation(cumulative_values2, board_val, k, idx3, game, np.array([-1]))

                if break_y == BreakState.BREAK_YES:  # Either a full-column (worst value given) or a win (best one given).
                    tree.child[i].child[j].child[k].value = cumulative_values3
                    continue
                idx4 = []

                for v in range(0, 7):  # other player plays
                    old_board3 = old_board2.copy()

                    game, board_val = assign_weight(old_board3, v, other_player, BOARD_VALUES)
                    break_y, cumulative_values4 = heuristic_evaluation(cumulative_values3, board_val, v, idx4, game)

                    # Last layers' nodes assigned the top-down cumulative heuristic value.
                    tree.child[i].child[j].child[k].child[v].value = cumulative_values4

                _, val_4 = min_child(tree.child[i].child[j].child[k], idx4)
                tree.child[i].child[j].child[k].value = val_4  # Assigning the value to father of minimal node.
            _, val_3 = max_child(tree.child[i].child[j], idx3)
            tree.child[i].child[j].value = val_3  # Assigning the value to father of maximal node.
        _, val_2 = min_child(tree.child[i], idx2)
        tree.child[i].value = val_2  # Assigning the value to father of minimal node.
    action, tree.value = max_child(tree, idx1)

    action = PlayerAction(action)  # Action to be taken in the Class PlayerAction

    if start == 10:  # If it is the 1st movement, 1st action performed is the optimal: column 3.
        action = 3

    saved_state_out = tree

    return action, saved_state_out