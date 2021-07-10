import numpy as np
from agents.common import GameState, BoardPiece, PlayerAction, check_end_state, apply_player_action
from agents.agents_random.random import generate_move_random


class TreeNode:
    """
    Tree structure class to store all nodes and their corresponding values.

    ...

    Attributes
    ----------
    board : np.array
        state of the board (matrix)
    move: PlayerAction
        move performed on the board in that turn
    parent: TreeNode or None
        parent node in the hierarchy
    child: TreeNode or None
        child nodes in the hierarchy
    turn_player: BoardPiece
        player who played this turn (last move on board).
    total_games: int
        number of games in which this node or its children have performed a random simulation (default is 0)
    wins: int
        number of wins in which this node or its children have won in the random simulation (default is 0)
    winner: bool
        whether this node is a winning node (winning combination of pieces already in the board) (default is False)
    loser: bool
        whether this node is a losing node (losing combination of pieces already in the board) (default is False)
    terminal: bool
        whether this is a terminal node (losing/winning/draw) (default is False)
    is_root: bool
        whether this node is the is_root of this stage of the game
    Methods
    -------
    new_child(children, main_player, last_action)
        Appending new children to the tree, using self as parent
    """

    def __init__(self, board, move, parent, turn_player):
        """ Initialization of the Tree assigning the attributes to the node.

        Parameters
        ----------
        board : np.array
            State of the board (matrix)
        parent: TreeNode or None
            Parent node in the hierarchy
        move: PlayerAction
            Move performed on the board in that turn
        turn_player: BoardPiece
            Player who played this turn (last move on board)
        """

        self.board = board
        self.parent = parent
        self.turn_player = turn_player
        self.move = move

        self.child = None
        self.total_games = 0
        self.wins = 0
        self.winner = False
        self.loser = False
        self.terminal = False
        self.is_root = False

    def new_child(self, children, main_player):
        """Appending new children to the tree, using self as the parent.

        Parameters
        ----------
        children: TreeNode
            new TreeNode structure to be appended as a child to self
        main_player: BoardPiece
            piece of the player using this tree node (machine_player)
        """
        # Appending the new child and choosing it as node variable
        if self.child is None:
            self.child = []
        self.child.append(children)
        node = self.child[-1]

        # Checking whether it is a terminal node (result different from
        # GameState.STILL_PLAYING and if so, load it into its attributes
        result = check_end_state(node.board, node.turn_player, node.move)

        if main_player == node.turn_player and result == GameState.IS_WIN:
            node.winner = True
            node.terminal = True
        elif main_player != node.turn_player and result == GameState.IS_WIN:
            node.loser = True
            node.terminal = True
        elif result == GameState.IS_DRAW:
            node.terminal = True

    def find_ucb1(self):
        """Finding the node's upper confidence bound (UBC1).

        Returns
        -------
        ucb1: None or float
            if a game has been played by the node already, return the ucb1 value, otherwise return None
        """
        ucb1 = None

        if self.total_games == 0:
            return ucb1
        else:
            # sqrt(2) used as exploration parameter
            ucb1 = self.wins / self.total_games + np.sqrt(2 * np.log(self.parent.total_games) / self.total_games)
            return ucb1

    def find_best_child(self):
        """Finding the child of a node with best UCB value.

        Returns
        -------
        node: TreeNode
            leaf node with highest UCB values
        """
        node = self
        ucb_values = [children.find_ucb1() for children in node.child]

        # Choosing random children if there is a child which has not been
        # explored yet (meaning it has None as its UCB value)
        if None in ucb_values:
            node = node.child[np.random.randint(len(node.child))]
        else:
            node = node.child[np.argmax(ucb_values)]

        return node

    def select_node(self):
        """Performing the SELECTION step.

        Finding the last node that is not terminal and has no children yet (leaf node),
        going through a path of selection guided by the UCB values of the nodes.

        Returns
        -------
        node: TreeNode
            leaf node with highest parental UCB values
        """
        node = self

        while node.child is not None and not node.terminal:

            node = node.find_best_child()

        return node

    def check_winning_children(self):
        """Check whether any children node has a board winning combination.

        Returns
        -------
        winners: list
            indexes of the winners which have a winning combination
        """
        booleans = np.empty(len(self.child))
        for i, children in enumerate(self.child):
            booleans[i] = children.winner
        winners = np.argwhere(booleans)

        # Return an empty list if no winners among the children
        if len(winners) == 0:
            return []
        else:
            return winners

    def check_losing_children(self):
        """Check whether any children node has a board losing combination.

        Returns
        -------
        losers: list
            indexes of the winners which have a losing combination
        """

        booleans = np.empty(len(self.child))
        for i, children in enumerate(self.child):
            booleans[i] = children.loser
        losers = np.argwhere(booleans)

        # Return an empty list if no losers among the children
        if len(losers) == 0:
            return []
        else:
            return losers

    def opponent_choice(self, action):

        moves = [children.move for children in self.child]
        ind_opponent_move = int(np.argwhere(moves == action))
        child_opponent = self.child[ind_opponent_move]

        return child_opponent

    def expansion(self, main_player):

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
                                   self.turn_player)), main_player)
            else:
                board = old_board.copy()
                apply_player_action(board, PlayerAction(cols), player=change_player(self.turn_player))
                self.new_child(TreeNode(board, PlayerAction(cols), parent=self, turn_player=change_player(
                               self.turn_player)), main_player)

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

        return node, win


def back_prop(node, winning):

    parent = node

    while parent.is_root is False:
        parent = parent.parent
        parent.total_games += 1

        if winning:
            parent.wins += 1


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
