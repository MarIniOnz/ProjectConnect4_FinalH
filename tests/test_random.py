import numpy as np
from agents.common import BoardPiece, PlayerAction

def test_random():
    from agents.agents_random.random import generate_move_random
    board = np.array([[1, 2, 2, 0, 1, 2, 2],
                      [2, 1, 1, 2, 1, 2, 2],
                      [2, 2, 1, 1, 1, 2, 2],
                      [2, 1, 2, 2, 2, 1, 1],
                      [1, 2, 1, 1, 1, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2]])
    action, saved_state = generate_move_random(board,BoardPiece(1),saved_state=0)

    assert isinstance(action,PlayerAction)
    assert action == PlayerAction(3)  #Taking the empty one