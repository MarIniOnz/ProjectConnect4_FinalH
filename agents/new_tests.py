import numpy as np
from typing import Optional
from enum import Enum
from agents.common import initialize_game_state, GameState, BoardPiece, PlayerAction, SavedState, check_end_state, apply_player_action
from agents.agents_random.random import generate_move_random



def change_player(player: PlayerAction):

    if player == BoardPiece(1):  # Finding out which player is who.
        player = BoardPiece(2)
    elif player == BoardPiece(2):
        player = BoardPiece(1)

    return player

# def check_players(player):
#
#     if player == BoardPiece(1):  # Finding out which player is who.
#         other_player = BoardPiece(2)
#     elif player == BoardPiece(2):
#         other_player = BoardPiece(1)
#     else:
#         raise('Error in player input')
#
#     return other_player

def check_win(turn_player,player):
    if turn_player == player:
        return True
    else:
        return False


def win_game(board,player,turn_player):
    win = GameState.STILL_PLAYING
    # other_player = check_players(player)

    while win == GameState.STILL_PLAYING:
        move,_ = generate_move_random(board,turn_player,None)
        apply_player_action(board,move,turn_player)
        if check_end_state(board,turn_player) == GameState.IS_WIN:
            win = GameState.IS_WIN
        else:
            turn_player = change_player(turn_player)

    if check_win(turn_player,player): return 1
    else: return 0

board = initialize_game_state()
print(win_game(board,BoardPiece(1),BoardPiece(1)))