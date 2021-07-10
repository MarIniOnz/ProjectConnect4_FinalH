from agents.common import PlayerAction, BoardPiece, apply_player_action
import numpy as np


def generate_move_random(board, player, saved_state=None):
    """Random getting a board and the corresponding player turn and returning a non-full column.

    Yields a non-full column action to be performed considering the board state.

    Parameters
    ----------
    board: np.array
        current state of the board
    player: BoardPiece
        whose turn is it
    saved_state: None
        not needed for a random generator

    Returns
    -------
    action: PlayAction
        Column to use.
    saved_state: None.
        not needed for a random generator
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
