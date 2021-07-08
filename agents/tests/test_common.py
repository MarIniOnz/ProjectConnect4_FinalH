import numpy as np
from agents.common import GameState, BoardPiece,PlayerAction

NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece


def test_initialize_game_state():
    from agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)

    """
    assert CONDITON , "OutputString"
    """

def test_pretty_print_board():
    from agents.common import pretty_print_board,initialize_game_state

    board = initialize_game_state()
    board_str = pretty_print_board(board)
    nlines = 9

    assert len(board_str.splitlines()) == nlines
    assert board_str[-1] == '|'
    assert board_str[0] == '|'
    assert isinstance(board_str,str)
    # Remark: you should check specific examples, this test does not mean the conversion works at all


def test_string_to_board():
    from agents.common import string_to_board
    from agents.common import pretty_print_board, initialize_game_state

    board = initialize_game_state()
    board_str = pretty_print_board(board)
    ret = string_to_board(board_str)

    assert isinstance(ret, np.ndarray)
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)
    # Remark: again, test specific examples. This test also depends on pretty_print_board, which  might produce circular dependencies

def test_apply_player_action():
    from agents.common import apply_player_action,initialize_game_state

    board = initialize_game_state()
    board[5, 0] = PLAYER2
    board[5, 1] = PLAYER1
    board[5, 2] = PLAYER2
    board[5, 3] = PLAYER1
    board[5, 4] = PLAYER1
    board[5, 5] = PLAYER1

    copy_board = board.copy()
    old_board, position = apply_player_action(board,PlayerAction(3),PLAYER1,True,True)

    assert old_board.all() ==  copy_board.all()
    assert position == (4,3)
    assert board[position] == PLAYER1

    board[:,0] = PLAYER1
    position2 = apply_player_action(board, PlayerAction(0), PLAYER1, False, True)
    assert position2 == 0  # Return 0 if full column.

def test_connected_four():
    from agents.common import connected_four, initialize_game_state

    board = initialize_game_state()
    assert not connected_four(board, PLAYER2)
    board[2:6,0] = PLAYER1
    assert connected_four(board,PLAYER1)
    board = initialize_game_state()
    board[2:6, 0] = PLAYER2
    assert connected_four(board, PLAYER2)

def test_check_end_state():
    from agents.common import check_end_state, initialize_game_state

    board = initialize_game_state()
    assert check_end_state(board,PLAYER1) == GameState.STILL_PLAYING
    board[2:6, 0] = PLAYER1
    assert check_end_state(board,PLAYER1) == GameState.IS_WIN
    board = np.array([[1, 2, 2, 1, 1, 2, 2],
                      [2, 1, 1, 2, 1, 2, 2],
                      [2, 2, 1, 1, 1, 2, 2],
                      [2, 1, 2, 2, 2, 1, 1],
                      [1, 2, 1, 1, 1, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2]])

    assert check_end_state(board,PLAYER1) == GameState.IS_DRAW