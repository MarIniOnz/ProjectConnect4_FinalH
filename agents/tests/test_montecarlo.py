import numpy as np
from agents.common import GameState, BoardPiece,PlayerAction

def test_Tree_Node():
    from agents.agent_Monte_Carlo.montecarlo import Tree_Node

    # board = Init
    # tree = Tree_Node()

def test_column_free():
    from agents.agent_Monte_Carlo.montecarlo import column_free
    board = np.array([[1, 2, 2, 0, 1, 2, 2],
                      [2, 1, 1, 0, 1, 2, 2],
                      [2, 2, 1, 1, 1, 2, 2],
                      [2, 1, 2, 2, 2, 1, 1],
                      [1, 2, 1, 1, 1, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2]])

    assert column_free(board,3)

def test_valid_columns():
    from agents.agent_Monte_Carlo.montecarlo import valid_columns

    board = np.array([[1, 2, 2, 0, 1, 2, 2],
                      [2, 1, 1, 2, 1, 2, 2],
                      [2, 2, 1, 1, 1, 2, 2],
                      [2, 1, 2, 2, 2, 1, 1],
                      [1, 2, 1, 1, 1, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2]])

    print(valid_columns(board))
    assert valid_columns(board) == 3

    board = np.array([[1, 2, 0, 0, 1, 2, 2],
                      [2, 1, 1, 2, 1, 2, 2],
                      [2, 2, 1, 1, 1, 2, 2],
                      [2, 1, 2, 2, 2, 1, 1],
                      [1, 2, 1, 1, 1, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2]])

    assert 2,3 in valid_columns(board)

    board = np.ones((6,7))

    assert valid_columns(board) == None
