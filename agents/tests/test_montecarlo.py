import numpy as np
from agents.common import GameState, BoardPiece,PlayerAction

def test_Tree_Node():
    from agents.agent_Monte_Carlo.montecarlo import Tree_Node

    # board = Init
    # tree = Tree_Node()

def test_valid_columns():
    from agents.agent_Monte_Carlo.montecarlo import valid_columns

    board = np.array([[1, 2, 2, 0, 1, 2, 2],
                      [2, 1, 1, 2, 1, 2, 2],
                      [2, 2, 1, 1, 1, 2, 2],
                      [2, 1, 2, 2, 2, 1, 1],
                      [1, 2, 1, 1, 1, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2]])

    assert valid_columns(board) == 3

    board = np.array([[1, 2, 0, 0, 1, 2, 2],
                      [2, 1, 1, 2, 1, 2, 2],
                      [2, 2, 1, 1, 1, 2, 2],
                      [2, 1, 2, 2, 2, 1, 1],
                      [1, 2, 1, 1, 1, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2]])

    assert valid_columns(board) == np.array([2,3])

def test_change_player(player:)