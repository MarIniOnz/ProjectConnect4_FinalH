from agents.common import PlayerAction, BoardPiece, SavedState, GenMove, apply_player_action
import numpy as np
from typing import Optional


def generate_move_random(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]):
    """Random getting a board and the corresponding player turn and returning
          a non-full column.

       Yields a non-full column action to be performed considering the board state.
       Args:
           board: Current state of the board
           player: Whose turn is it.
           saved_state: Optimal pre-computation work performed in previous steps.

       Returns:
           action: Column to use.
           saved_state: Not-yet implemented, but needed for the main.py algorithm.

       """

    exit_yes = False
    old_board = board.copy()
    action = np.array([0])

    while not exit_yes:
        action = PlayerAction(np.random.randint(7))
        old_board, position = apply_player_action(old_board, action, player, True, True)
        if position is not None:
            exit_yes = True
    
    return action, saved_state