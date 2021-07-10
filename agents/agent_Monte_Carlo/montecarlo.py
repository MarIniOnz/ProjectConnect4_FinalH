import numpy as np
from agents.common import GameState, BoardPiece, PlayerAction, check_end_state, apply_player_action
from agents.agents_random.random import generate_move_random


class TreeNode:
    """
    Tree structure class to store all nodes and their corresponding values.

    Attributes
    ----------
    board: np.array
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

    Methods
    -------
    new_child(children, main_player, last_action)
        Appending new children to the tree, using self as parent
    find_ucb1()
        Finding the node's upper confidence bound (UBC1)
    find_best_child():
        Finding the child of a node with best UCB value
    select_node():
        Performing the SELECTION step
    check_winning_children():
        Check whether any children node has a board winning combination
    check_losing_children():
        Check whether any children node has a board losing combination
    opponent_choice(action):
        Take the action made by the opponent and find which child of the node corresponds to that case
    losing_case(losing_nodes):
        Prevent losing scenarios by returning lose-preventing nodes
    expansion(main_player):
        Performs the EXPANSION of the algorithm
    """

    def __init__(self, board, move, parent, turn_player):
        """ Initialization of the Tree assigning the attributes to the node.

        Parameters
        ----------
        board: np.array
            state of the board (matrix)
        parent: TreeNode or None
            parent node in the hierarchy
        move: PlayerAction
            move performed on the board in that turn
        turn_player: BoardPiece
            player who played this turn (last move on board)
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
        """Take the action made by the opponent and find which child of the node
           corresponds to that case.

        Parameters
        ----------
        action: PlayerAction
            action performed by the opponent

        Returns
        -------
        child_opponent: TreeNode
            child of the node corresponding to that move of the opponent
        """

        moves = [children.move for children in self.child]
        ind_opponent_move = int(np.argwhere(moves == action))
        child_opponent = self.child[ind_opponent_move]

        return child_opponent

    def losing_case(self, losing_nodes):
        """Prevent losing scenarios by returning lose-preventing nodes.

        Take the action would make the opponent win, returns the node
        that would prevent that from happening and makes the node that
        would end up in the situation as a terminal, loser node.

        Parameters
        ----------
        losing_nodes: list
            indexes of the winners which have a losing combination

        Returns
        -------
        node: TreeNode
            node preventing the loss from happening.
        """
        move_needed = self.child[int(losing_nodes[0])].move
        node = self.parent.opponent_choice(move_needed)

        # This node is thought to be terminal as it is assumed the opponent will
        # choose the winning move.
        self.terminal = True
        self.loser = True
        self.wins = 0

        return node

    def expansion(self, main_player):
        """Performs the EXPANSION of the algorithm.

        Expands the tree from the self node if it is non-terminal by creating the children with
        possible moves and detecting whether among them there are losing or winning nodes.
        If there is a wining child, selects that node for the back propagation, if there is a losing
        node, it selects the node preventing that lost and if there are none, performs a randomized
        simulation using a random children node's board as start-point

        Parameters
        ----------
        main_player: BoardPiece
            piece of the player using this tree node (machine_player)

        Returns
        -------
        node: TreeNode
            node that performed the randomized simulation or a winning/preventing from losing node
        win: bool
            whether the outcome of the game of that node was a win
        """
        cols = valid_columns(self.board)  # Finding all possible moves
        win = False
        node = self

        # Check whether it is possible to create new children
        if not self.terminal and cols is not None:

            old_board = self.board.copy()
            if cols.size > 1:
                for i, column in enumerate(cols):
                    board = old_board.copy()
                    # Create a child for each possible move
                    apply_player_action(board, column, player=change_player(self.turn_player))
                    self.new_child(TreeNode(board, PlayerAction(column), parent=self, turn_player=change_player(
                        self.turn_player)), main_player)

            else:  # in case there is only 1 column (had problems with this scenario, so i separated it)
                board = old_board.copy()
                apply_player_action(board, PlayerAction(cols), player=change_player(self.turn_player))
                self.new_child(TreeNode(board, PlayerAction(cols), parent=self, turn_player=change_player(
                    self.turn_player)), main_player)

            # Check whether there are any winning or losing children
            winning_nodes = self.check_winning_children()
            losing_nodes = self.check_losing_children()

            # Both scenarios cannot happen at the same time, since both a loss and a win
            # cannot happen at the same turn and winning/losing nodes are terminal

            # If there are losing children, find the node that prevents it and treat it
            # as a win (biasing the algorithm towards desiring this preventing node)
            if len(losing_nodes) > 0:
                node = self.losing_case(losing_nodes)
                win = True

            # If there are winning children, use them as the back propagating node so that
            # system orients towards this node
            elif len(winning_nodes) > 0:
                node = self.child[int(winning_nodes[0])]
                win = node.winner

            # If there are no winning nor losing nodes, pick a random children and simulate a
            # a random game from that board state on.
            else:
                ind = int(np.random.randint(cols.size))
                node = self.child[ind]
                old_board = node.board.copy()
                if node.terminal is False:
                    win = random_game(node.board, main_player, node.turn_player)
                node.board = old_board

        # Check whether it is terminal because it is a winning or a losing node
        elif self.terminal and cols is not None:
            if node.winner:
                win = node.winner

        # If cols are None, it should be a terminal node
        elif not self.terminal and cols is None:
            self.terminal = True

        return node, win


def back_prop(node, winning):
    """Performs the BACKPROPAGATION of the algorithm.

    Propagates the results of simulations from lower nodes to higher in the hierarchy,
    parental to them, as well as increasing the total games per node as back-propagating
    occurs.

    Parameters
    ----------
    node: TreeNode
        node that performed the randomized simulation or a winning/preventing from losing node
    winning: bool
        whether the outcome of the game of that node was a win
    """
    parent = node

    while parent is not None:
        parent.total_games += 1

        if winning:
            parent.wins += 1

        parent = parent.parent


def column_free(board, column) -> bool:
    """Check whether the column of a board is free.

    Parameters
    ----------
    board: np.array
        state of the board (matrix)
    column: int
        column to be checked in the board

    Returns
    -------
    free: bool
        whether the column is free to be used or not
    """
    free = (sum(board[:, column] == 0) - 1) >= 0

    return free


def valid_columns(board):
    """Check which are the valid columns for moves.

    Parameters
    ----------
    board: np.array
        state of the board (matrix)

    Returns
    -------
    ind: np.array or None
        columns that can be used
    """
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
        ind = np.array(ind)
        return ind
    else:
        return ind


def change_player(player) -> BoardPiece:
    """Changes the player given the current one.

    Parameters
    ----------
    player: BoardPiece
        given player

    Returns
    -------
    other_player: BoardPiece
        the other player
    """
    if player == BoardPiece(1):
        other_player = BoardPiece(2)
    else:
        other_player = BoardPiece(1)

    return other_player


def same_player(turn_player, main_player):
    """Checks whether the turn player is the same one as the main one.

    Parameters
    ----------
    turn_player: BoardPiece
        player of the turn
    main_player: BoardPiece
        main player

    Returns
    -------
    same: bool
        whether the player of this turn is the same as the main one
    """
    same = False
    if turn_player == main_player:
        same = True

    return same


def random_game(board, main_player, turn_player):
    """Performs a randomized game.

    A random game is performed from the board initial position until a final
    position is reached. Then, it is checked whether on that final position,
    the main player won or not (draw or loss).

    Parameters
    ----------
    board: np.array
        state of the board (matrix)
    main_player: BoardPiece
        main player
    turn_player: BoardPiece
        player of the turn

    Returns
    -------
    win: bool
        whether the main player won in the simulation
    """
    state = GameState.STILL_PLAYING
    win = False

    while state == GameState.STILL_PLAYING:
        move, _ = generate_move_random(board, turn_player, None)
        apply_player_action(board, move, turn_player)

        # Checking whether the game has come to an end
        if check_end_state(board, turn_player, move) == GameState.IS_WIN:
            state = GameState.IS_WIN
        elif check_end_state(board, turn_player, move) == GameState.IS_DRAW:
            state = GameState.IS_DRAW
        else:
            turn_player = change_player(turn_player)  # Next turn, change of players.

    # There is only a win if the player that won in the last turn is the same as the main player.
    if same_player(turn_player, main_player) and state == GameState.IS_WIN:
        win = True

    return win
