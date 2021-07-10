import numpy as np
import time
from agents.common import BoardPiece, apply_player_action, PlayerAction
from agents.agent_Monte_Carlo.montecarlo import TreeNode, change_player, back_prop


def montecarlo(board: np.ndarray, player: BoardPiece, saved_state, last_action: PlayerAction, train_time=5):
    # Time counter for the while loop.
    # train_time = 3
    # n = 2000
    num = 0

    if last_action is None:
        action = PlayerAction(3)
        apply_player_action(board, action, player, copy=False, pos=False)
        root = TreeNode(board, action, None, player)
        node = root.select_node()
        node.expansion(player)
        saved_state = root

    else:
        if saved_state is None:
            root = TreeNode(board, last_action, None, change_player(player))

        else:
            try:
                root = saved_state.opponent_choice(last_action)
            except:
                root = TreeNode(board, last_action, None, change_player(player))

        start = int(round(time.time()))
        present = int(round(time.time()))

        while (present - start) < train_time:
            node = root.select_node()
            node, win = node.expansion(player)
            back_prop(node, win)
            present = int(round(time.time()))
            num += 1

        best_child = root.find_best_child()
        saved_state = best_child
        action = best_child.move

    print(root.total_games)

    return action, saved_state
