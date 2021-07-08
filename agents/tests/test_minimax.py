import numpy as np
from agents.common import GameState, BoardPiece,PlayerAction

NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

BOARD_VALUES = np.array([[3, 4, 5, 7, 5, 4, 3],
                         [4, 6, 8, 10, 8, 6, 4],
                         [5, 8, 11, 13, 11, 8, 5],
                         [5, 8, 11, 13, 11, 8, 5],
                         [4, 6, 8, 10, 8, 6, 4],
                         [3, 4, 5, 7, 5, 4, 3]])


def test_minmax_tree():
    from agents.agent_minimax.minimax import initialization_minimax_tree
    from agents.agent_minimax.minimax import Tree

    tree = initialization_minimax_tree()
    assert isinstance(tree,Tree)
    assert tree.value == -10000
    assert tree.child[0].value == -10000
    assert tree.child[0].child[1].value == 10000

def test_assign_weight():
    from agents.agent_minimax.minimax import assign_weight
    # Remark: another sign of too complicated functions is that you have to test for multiple things.
    board = np.array([[1, 2, 2, 0, 1, 2, 2],
                      [2, 1, 1, 2, 1, 2, 2],
                      [2, 2, 1, 1, 1, 2, 2],
                      [2, 1, 2, 2, 2, 1, 1],
                      [1, 2, 1, 1, 1, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2]])

    game,val_board = assign_weight(board, 3, PLAYER1, board_values=BOARD_VALUES)
    assert val_board == BOARD_VALUES[0,3]
    assert game == GameState.IS_DRAW

    board = np.array([[1, 2, 2, 0, 1, 2, 2],
                      [2, 1, 1, 2, 1, 2, 2],
                      [2, 2, 1, 1, 1, 2, 2],
                      [2, 1, 2, 2, 2, 1, 1],
                      [1, 2, 1, 1, 1, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2]])

    game, val_board = assign_weight(board, 2, PLAYER1, board_values=BOARD_VALUES) # Full column
    assert val_board == 100 # Indicating that we are running into full-column

def test_eval_heu():
    from agents.agent_minimax.minimax import heuristic_evaluation, assign_weight

    board = np.array([[1, 0, 0, 0, 1, 2, 2],
                      [2, 1, 1, 2, 1, 2, 2],
                      [2, 2, 1, 1, 1, 2, 2],
                      [2, 1, 2, 2, 2, 1, 1],
                      [1, 2, 1, 1, 1, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2]])

    cumulative_heuristic = 10
    i = 3
    idx = []

    game, val_board = assign_weight(board, 3, PLAYER1, board_values=BOARD_VALUES)
    break_y, minimizing_case= heuristic_evaluation(cumulative_heuristic, val_board, i, idx, game,
                                                   node_type=np.array([1]))

    assert minimizing_case == cumulative_heuristic - val_board # Proving if cumulative works
    assert idx == [i]

    break_y, maximizing_case = heuristic_evaluation(cumulative_heuristic, val_board, i, idx, game,
                                                    node_type=np.array([-1]))
    assert idx == [i,i]
    assert maximizing_case == cumulative_heuristic + val_board  # Proving if cumulative works
    assert not break_y # PLAYER1 did not win and no full-column

    game, val_board = assign_weight(board, 5, PLAYER1, board_values=BOARD_VALUES)
    break_y, column_full = heuristic_evaluation(cumulative_heuristic, val_board, i, idx, game, node_type=np.array([-1]))

    assert column_full == -100000
    assert break_y # We break because we do not want to explore more into a
                   # situation in which we are putting into a full-column a piece.
    # Remark: way too complicated!

def test_max_child():
    from agents.agent_minimax.minimax import max_child, initialization_minimax_tree

    tree = initialization_minimax_tree()
    index = []
    for i in range(0,7):
        index.append(i)
        tree.child[i].value = i

    idx, maxi = max_child(tree,index)
    assert idx == i
    assert maxi == i

def test_min_child():
    from agents.agent_minimax.minimax import min_child, initialization_minimax_tree

    tree = initialization_minimax_tree()
    index = []
    for i in range(0,7):
        index.append(i)
        tree.child[i].value = i

    idx, mini = min_child(tree,index)
    assert idx == 0
    assert mini == 0

def test_minimax_action():
    from agents.agent_minimax.minimax import minimax_action

    board = np.array([[1, 0, 0, 0, 0, 2, 2],
                      [2, 1, 0, 2, 0, 2, 2],
                      [2, 2, 1, 1, 1, 2, 2],
                      [2, 1, 1, 2, 1, 1, 1],
                      [1, 2, 2, 1, 1, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2]])

    action, tree = minimax_action(board,PLAYER1, [])

    assert action == 4  # Winning.
    assert tree.value == 10000  # Heuristics of winning.

    board = np.array([[1, 0, 0, 0, 0, 2, 1],
                      [2, 1, 0, 2, 0, 1, 2],
                      [2, 2, 1, 1, 0, 2, 2],
                      [2, 1, 1, 2, 2, 1, 1],
                      [1, 2, 2, 1, 2, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2]])

    action, tree = minimax_action(board, PLAYER1, [])

    assert action == 4 # Preventing win.

    board[:] = 0

    action, tree = minimax_action(board, PLAYER2, [])

    assert action == 3 # Starting with the very nice central slot.