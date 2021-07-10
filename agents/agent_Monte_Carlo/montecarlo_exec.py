import time
from agents.common import BoardPiece, apply_player_action, PlayerAction
from agents.agent_Monte_Carlo.montecarlo import TreeNode, change_player, back_prop


def montecarlo(board, player, saved_state, last_action, train_time=5):
    """ Performance of the MonteCarlo algorithm.

    A MonteCarlo algorithm is performed for as long as the given training time.

    Parameters
    ----------
    board: np.array
        state of the board (matrix)
    player: BoardPiece
        player that performs the MonteCarlo algorithm
    saved_state: TreeNode or None
        previously chosen node
    last_action: BoardPiece or None
        action performed by the other player in previous turn
    train_time: int
        time devoted for the MonteCarlo algorithm

    Returns
    -------
    action: PlayerAction
        selected action for the game
    saved_state: TreeNode
        chosen node from which is action was extracted
    """
    # If the agent starts the game, the column 3 is chosen (best choice)
    if last_action is None:
        action, saved_state = blank_board(board, player)

    # If not, the root is established, taken into consideration which one was the move of the opponent
    else:
        root = establish_root(board, player, saved_state, last_action)

        start = int(round(time.time()))
        present = int(round(time.time()))

        while (present - start) < train_time:  # Loop until the training time is over
            node = root.select_node()  # SELECTION
            node, win = node.expansion(player)  # EXPANSION
            back_prop(node, win)  # BACKPROPAGATION

            # Time update
            present = int(round(time.time()))

        # The child with the best UCB value from the root is chosen, along with its action
        best_child = root.find_best_child()
        saved_state = best_child
        action = best_child.move

    return action, saved_state


def blank_board(board, player: BoardPiece):
    """ Initialization of the root of the tree and assignment of the column 3 as first action.

    Parameters
    ----------
    board: np.array
        state of the board (matrix)
    player: BoardPiece
        player that performs the MonteCarlo algorithm

    Returns
    -------
    action: PlayerAction
        selected action for the game (column 3)
    saved_state: TreeNode
        chosen node from which is action was extracted
    """
    action = PlayerAction(3)
    apply_player_action(board, action, player, copy=False, pos=False)
    root = TreeNode(board, action, None, player)

    # Creating its child nodes
    node = root.select_node()
    node.expansion(player)
    saved_state = root

    return action, saved_state


def establish_root(board, player, saved_state, last_action):
    """ Initialization of the root of the tree and assignment of the column 3 as first action.

    Parameters
    ----------
    board: np.array
        state of the board (matrix)
    player: BoardPiece
        player that performs the MonteCarlo algorithm
    saved_state: TreeNode or None
        previously chosen node
    last_action: BoardPiece
        action performed by the other player in previous turn

    Returns
    -------
    root: TreeNode
        node that will serve as the root for this turn
    """
    if saved_state is None:  # Initialization in case of first turn
        root = TreeNode(board, last_action, None, change_player(player))

    else:
        try:
            root = saved_state.opponent_choice(last_action)  # The node whose action was selected by opponent
            root.parent = None  # New root of the algorithm, no parents needed
        except:
            root = TreeNode(board, last_action, None, change_player(player))  # In case no children were found
            root.parent = None

    return root
