import numpy as np
import time
from agents.common import BoardPiece
from agents.agent_Monte_Carlo.montecarlo import Tree_Node

def montecarlo(board: np.ndarray, player: BoardPiece, saved_state):

    if saved_state == None:
        saved_state = Tree_Node(board, None, None, player)

    # Time counter for the while loop.
    start = int(round(time.time()))
    present = start
    train_time = 5

    root = saved_state
    first = False

    if sum(sum(board[:, :]) == 0) == 7:
        first = True


    while (present-start) < train_time and not first:

        node = root.select_node()
        node.expansion_and_back_prop(player)
        present = int(round(time.time()))

    if first:
        action = 3
    else:
        action = root.select_node().move

    saved_state = root.select_node()

    return action, saved_state



