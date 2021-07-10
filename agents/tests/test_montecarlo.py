import numpy as np
from agents.common import BoardPiece, PlayerAction, initialize_game_state, apply_player_action
import time


# Montecarlo
def test_TreeNode():
    from agents.agent_Monte_Carlo.montecarlo import TreeNode, change_player, back_prop

    board = initialize_game_state()
    player = BoardPiece(1)
    move = PlayerAction(3)

    # Testing TreeNode Initialization
    node = TreeNode(board, move, None, player)
    board = node.board.copy()

    assert node.move == PlayerAction(3)
    assert node.turn_player == player
    assert node.parent is None
    assert node.child is None

    # Testing new_child method.
    apply_player_action(board, PlayerAction(2), player=change_player(player))
    node.new_child(TreeNode(board, PlayerAction(2), parent=node, turn_player=change_player(player)), player)

    node_new = node.child[0]

    assert node_new.move == PlayerAction(2)
    assert node_new.parent == node
    assert node_new.turn_player == BoardPiece(2)

    # Testing check_winning_children and check_losing_children
    lost_child = node.check_losing_children()
    won_child = node.check_winning_children()

    assert len(lost_child) == 0
    assert len(won_child) == 0

    # Testing back_prop.
    assert node_new.total_games == 0
    back_prop(node_new, False)
    assert node_new.total_games == 1
    back_prop(node_new, True)
    assert node_new.total_games == 2
    assert node_new.wins == 1

    # Testing find_ucb1
    ucb = node_new.find_ucb1()
    assert ucb == node_new.wins / node_new.total_games + \
           np.sqrt(2 * np.log(node_new.parent.total_games) / node_new.total_games)

    # Testing find_best_child
    best_child = node.find_best_child()
    assert best_child == node_new

    # Testing select_node
    select_node = node.select_node()
    assert select_node == node_new

    # Testing expansion
    board = initialize_game_state()
    player = BoardPiece(1)
    move = PlayerAction(3)

    # Testing TreeNode Initialization
    node = TreeNode(board, move, None, player)
    expanded_node, win = node.expansion(player)

    assert expanded_node.parent == node
    assert win is False or True
    assert expanded_node.winner is False and expanded_node.loser is False


def test_column_free():
    from agents.agent_Monte_Carlo.montecarlo import column_free

    board = np.array([[1, 2, 2, 0, 1, 2, 2],
                      [2, 1, 1, 0, 1, 2, 2],
                      [2, 2, 1, 1, 1, 2, 2],
                      [2, 1, 2, 2, 2, 1, 1],
                      [1, 2, 1, 1, 1, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2]])
    assert column_free(board, 3)


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
    assert 2, 3 in valid_columns(board)

    board = np.ones((6, 7))
    assert valid_columns(board) is None


def test_change_player():
    from agents.agent_Monte_Carlo.montecarlo import change_player

    assert BoardPiece(1) == change_player(BoardPiece(2))
    assert BoardPiece(2) == change_player(BoardPiece(1))


def test_same_player():
    from agents.agent_Monte_Carlo.montecarlo import same_player

    player = BoardPiece(1)

    assert same_player(BoardPiece(1), player) is True
    assert same_player(BoardPiece(2), player) is False


def test_random_game():
    from agents.agent_Monte_Carlo.montecarlo import random_game

    board = initialize_game_state()
    main_player = turn_player = BoardPiece(1)

    win = random_game(board, main_player, turn_player)

    assert win is False or True  # It can either lose/draw or win at the end of game.


# Montecarlo execution file.
def test_montecarlo():
    from agents.agent_Monte_Carlo.montecarlo_exec import montecarlo

    board = initialize_game_state()
    player = BoardPiece(1)

    action, saved_state = montecarlo(board, player, None, None)

    assert action == PlayerAction(3)  # First action if blank board os given is 3.
    assert action == PlayerAction(3)
    assert saved_state.parent is None
    assert saved_state.move == PlayerAction(3)
    assert saved_state.turn_player == BoardPiece(1)

    new_action = PlayerAction(2)
    board = saved_state.board.copy()
    apply_player_action(board, new_action, BoardPiece(2))

    start = int(round(time.time()))
    train_time = 5
    action, saved_state1 = montecarlo(board, player, saved_state, new_action, train_time)
    present = int(round(time.time()))
    time_used = present - start

    assert time_used == train_time
    assert saved_state1.child[0].move == PlayerAction(0)  # There are children, and all columns are open to be used.
    assert saved_state1.parent.total_games > 800  # At least 800 iterations.


def test_establish_root():
    from agents.agent_Monte_Carlo.montecarlo_exec import establish_root

    board = np.array([[1, 2, 0, 0, 1, 2, 2],
                      [2, 1, 1, 2, 1, 2, 2],
                      [2, 2, 1, 1, 1, 2, 2],
                      [2, 1, 2, 2, 2, 1, 1],
                      [1, 2, 1, 1, 1, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2]])
    last_action = PlayerAction(5)
    root = establish_root(board, BoardPiece(1), None, last_action)

    # The root is the node whose last move was the one executed by the opponent
    # player (turn_player == 2) and has no parenting node.
    assert root.move == last_action
    assert root.parent is None
    assert root.turn_player == BoardPiece(2)


def test_blank_board():
    from agents.agent_Monte_Carlo.montecarlo_exec import blank_board

    board = initialize_game_state()
    action, root = blank_board(board, BoardPiece(1))

    # First action is on the third column (best one)
    assert action == PlayerAction(3)
    assert root.parent is None
    assert root.move == PlayerAction(3)
    assert root.turn_player == BoardPiece(1)
