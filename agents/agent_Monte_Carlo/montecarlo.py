import numpy as np
from agents.common import GameState, BoardPiece, PlayerAction, check_end_state, apply_player_action
from agents.agents_random.random import generate_move_random


class TreeNode:

    def __init__(self, board, move, parent, turn_player):
        """ Initialization of the Tree assigning the value to the node."""
        self.board = board
        self.parent = parent
        self.child = None
        self.move = move
        self.total_games = 0
        self.wins = 0
        self.winner = False
        self.loser = False
        self.terminal = False
        self.turn_player = turn_player

    def new_child(self, children, main_player, last_action):
        if self.child is None:
            self.child = []
        self.child.append(children)  # Append not working
        node = self.child[-1]
        result = check_end_state(node.board, node.turn_player, last_action)

        if main_player == node.turn_player and result == GameState.IS_WIN:
            node.winner = True
            node.terminal = True
        elif main_player != node.turn_player and result == GameState.IS_WIN:
            node.loser = True
            node.terminal = True
        elif result == GameState.IS_DRAW:
            node.terminal = True

    def find_ucb1(self):
        if self.total_games == 0:
            return None
        else:
            ucb1 = self.wins / self.total_games + np.sqrt(2 * np.log(self.parent.total_games) / self.total_games)
            return ucb1

    def select_node(self):

        node = self

        while node.child is not None and not node.terminal:

            node = node.find_best_child()

        return node

    def find_best_child(self):

        node = self

        ucb_values = [children.find_ucb1() for children in node.child]
        if None in ucb_values:
            node = node.child[np.random.randint(len(node.child))]
        else:
            node = node.child[np.argmax(ucb_values)]

        return node

    def check_winning_children(self):

        booleans = np.empty(len(self.child))
        for i, children in enumerate(self.child):
            booleans[i] = children.winner
        winners = np.argwhere(booleans)

        if len(winners) == 0:
            return []
        else:
            return winners

    def check_losing_children(self):

        booleans = np.empty(len(self.child))
        for i, children in enumerate(self.child):
            booleans[i] = children.loser
        losers = np.argwhere(booleans)

        if len(losers) == 0:
            return []
        else:
            return losers

    def expansion_and_back_prop(self, main_player):

        cols = valid_columns(self.board)
        win = False
        node = self

        if not self.terminal and cols is not None:
            old_board = self.board.copy()
            if cols.size > 1:
                for i, column in enumerate(cols):
                    board = old_board.copy()
                    apply_player_action(board, column, player=change_player(self.turn_player))
                    self.new_child(TreeNode(board, PlayerAction(column), parent=self, turn_player=change_player(
                                   self.turn_player)), main_player, column)
            else:
                board = old_board.copy()
                apply_player_action(board, PlayerAction(cols), player=change_player(self.turn_player))
                self.new_child(TreeNode(board, cols, parent=self, turn_player=change_player(self.turn_player)),
                               main_player, cols)

            winning_nodes = self.check_winning_children()
            losing_nodes = self.check_losing_children()

            if len(losing_nodes) > 0:
                move_needed = self.child[int(losing_nodes[0])].move
                node = self.parent.opponent_choice(move_needed)
                # node.wins += node.parent.total_games * 20
                win = True
                # node.total_games += 1
                self.terminal = True
                self.wins = 0
                self.loser = True

            elif len(winning_nodes) > 0:
                ind = int(np.random.randint(cols.size))
                node = self.child[ind]
                win = node.winner

            else:
                ind = int(np.random.randint(cols.size))
                node = self.child[ind]
                old_board = node.board.copy()
                # start2 = time.time()
                win = win_game(node.board, main_player, node.turn_player)
                # print(f'Time: {time.time() - start2}')
                node.board = old_board

        elif self.terminal and cols is not None:
            if node.winner:
                win = node.winner

        elif not self.terminal and cols is None:
            self.terminal = True

        node.back_prop(win)

        return node

    def back_prop(self, winning):

        parent = self

        while parent is not None:
            parent.total_games += 1

            if winning:
                parent.wins += 1

            parent = parent.parent

    def opponent_choice(self, action):

        moves = [children.move for children in self.child]
        ind_opponent_move = int(np.argwhere(moves == action))
        child_opponent = self.child[ind_opponent_move]

        return child_opponent


def column_free(board, column):

    return (sum(board[:, column] == 0) - 1) >= 0


def valid_columns(board):
    ind = None
    valid = False

    for j in range(0, board.shape[1]):
        j_good = column_free(board, j)

        if j_good and not valid:
            ind = j
            valid = True
        elif j_good and valid:
            ind = np.hstack((ind, j))
    if ind is not None:
        return np.array(ind)
    else:
        return None


def change_player(player: PlayerAction):
    if player == BoardPiece(1):  # Finding out which player is who.
        player = BoardPiece(2)
    elif player == BoardPiece(2):
        player = BoardPiece(1)

    return player


def check_win(turn_player, player):
    if turn_player == player:
        return True
    else:
        return False


def win_game(board, player, turn_player):
    win = GameState.STILL_PLAYING

    while win == GameState.STILL_PLAYING:
        move, _ = generate_move_random(board, turn_player, None)
        apply_player_action(board, move, turn_player)
        if check_end_state(board, turn_player, move) == GameState.IS_WIN:
            win = GameState.IS_WIN
        elif check_end_state(board, turn_player, move) == GameState.IS_DRAW:
            win = GameState.IS_DRAW
        else:
            turn_player = change_player(turn_player)

    if check_win(turn_player, player) and win == GameState.IS_WIN:
        return True
    else:
        return False
