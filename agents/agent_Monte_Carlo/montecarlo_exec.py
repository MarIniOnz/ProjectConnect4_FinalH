import numpy as np
import time
from agents.common import BoardPiece, apply_player_action
from agents.agent_Monte_Carlo.montecarlo import Tree_Node

def montecarlo(board: np.ndarray, player: BoardPiece, saved_state, action):

    if saved_state == None:
        root = Tree_Node(board, None, None, player)

    # Time counter for the while loop.
    start = int(round(time.time()))
    train_time = 5
    n = 2000
    num = 0

    if action is None:
        action = 3
        apply_player_action(board, action, player=root.turn_player, copy=False, pos=False)
        root.board = board
        node = root.select_node()
        node.expansion_and_back_prop(player)
        saved_state = root

    else:

        root = saved_state.opponent_choice(action)

        start = int(round(time.time()))
        present = int(round(time.time()))

        # while (present-start) < train_time:
        while num < 500:

            node = root.select_node()
            node.expansion_and_back_prop(player)
            present = int(round(time.time()))
            num += 1


        best_child = root.find_best_child()
        saved_state = best_child
        action = best_child.move

    print(root.total_games)

    return action, saved_state



