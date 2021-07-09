from enum import Enum
from typing import Optional
import numpy as np
from typing import Callable, Tuple


BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = str(' ')
PLAYER1_PRINT = str('X')
PLAYER2_PRINT = str('O')

PlayerAction = np.int8  # The column to be played


class SavedState:
    """ Class used to save a computational result.

       Saving computational results so they can be used in further steps.

       Attributes:
           computational_result : Saved calculations.
    """
    def __init__(self, computational_result):
        self.computational_result = computational_result


class GameState(Enum):
    """ Class used to store the state of the game

       This creates the class so that it is stored whether a player has won, there is a draw
       or the game is still on.
   """
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def initialize_game_state() -> np.ndarray:
    """ Initializes C4 board.

    Returns an array, shape (6, 7) and data type (data type) BoardPiece, initialized to 0 (NO_PLAYER).
    Apparently the board is flipped here in comparison to the given convention.

    Returns:
        board: Initialized board with no pieces on it.
    """
    board = np.full((6, 7), NO_PLAYER)

    return board


def pretty_print_board(board: np.ndarray) -> str:
    """ Print a board as readable for the user.

    Returns `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout).
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |

    Args:
        board:Current state of the board (matrix).

    Returns:
        board_str: String to print the pretty readable board.
    """

    x, y = np.shape(board)

    board_str = '|'
    for p in range(0, y):
        board_str += '=='
    board_str += '|\n'

    for i in range(0, x):
        string = '|'

        for j in range(0, y):
            if board[i, j] == NO_PLAYER:
                string += NO_PLAYER_PRINT
            elif board[i, j] == PLAYER1:
                string += PLAYER1_PRINT
            elif board[i, j] == PLAYER2:
                string += PLAYER2_PRINT
            string += ' '
        board_str += string + '|\n'

    board_str += '|'
    for m in range(0, y):
        board_str += '=='
    board_str += '|\n'

    board_str += '|'
    for p in range(0, y):
        board_str += str(p) + ' '
    board_str += '|'

    return board_str


def string_to_board(pp_board: str) -> np.ndarray:
    """ String board to matrix values in an array.

    Takes the output of 'pretty_print_board' and turns it back into an array.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.

    Args:
        pp_board: String containing the pretty_print_board output.

    Returns:
        board_out: Board extracted from the string as a matrix with its associated BoardPieces.

    """
    start = 0
    board_out = np.ndarray([])
    for line in pp_board.split('\n'):
        if '=' in line:
            start += 1
        elif '=' not in line and start < 2:
            line = line[1:-2:2]
            print(line)
            line_np = np.array([])
            for j in range(0, len(line)):
                if line[j] == NO_PLAYER_PRINT:
                    line_np = np.append(line_np, NO_PLAYER)
                elif line[j] == PLAYER1_PRINT:
                    line_np = np.append(line_np, PLAYER1)
                elif line[j] == PLAYER2_PRINT:
                    line_np = np.append(line_np, PLAYER2)
            if line_np:
                board_out = np.hstack((board_out, line_np))

    board_out = board_out[1::]
    board_out = board_out.reshape(6, 7)

    return board_out


def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece,
        copy: bool = False, pos: bool = False
):
    """ Getting a new piece in the board corresponding to 'player' BoardPiece.

    In the 'board', a new piece is introduced in the column 'action' by 'player'.
    If the column is full, return 'position' = 0.

    Args:
        board: Current state of the board.
        action: Column in which the new piece will be introduced.
        player: Whose turn is it.
        copy: Whether a copy of the board is wanted as return.
        pos: Whether the position of the new piece is wanted as return
    Returns:
        old_board: Copy of the board before a new piece was introduced.
        position: Position of the new piece. Returns None if the column was full.

    """
    old_board = np.copy(board)
    row = np.sum(board[:, action] == 0) - 1
    position = None
    if row >= 0:
        board[row, action] = player
        position = row, action

    if copy and not pos:
        return old_board
    elif pos and not copy:
        return position
    elif pos and copy:
        return old_board, position
    else:
        return None


def findall(element: BoardPiece, board: np.array) -> np.ndarray:
    """ Checks whether there is an element in a matrix and if so, return their indexes.

    Args:
        element: Element to be found.
        board: 2D Board to be inspected.

    Returns:
        res: Result of the search, indexes of the 2D matrix in which the element is
             found.

    """
    result = []

    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == element:
                result.append([i, j])

    res = np.array(result)
    return res


def connected_four(board: np.ndarray, player: BoardPiece, last_action: PlayerAction = None) -> bool:
    """ Check if there are 4 connected pieces in the board for the player.

    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.

    Args:
        board: Current state of the board.
        player: Whose turn is it.
        last_action: Last action taken.
    Returns:
        bool: True if there are at least 4 connected pieces, False otherwise

    """
    if last_action is None:
        indexes = findall(player, board)
    else:
        indexes = np.array([[np.sum(board[:, last_action] == 0), last_action]])

    x, y = np.shape(board)

    dirs = np.array([[1, 1], [1, -1], [1, 0], [0, 1]])  # Directions to inspect the board.
    sums = np.zeros((indexes.shape[0], 4))

    for i in range(0, indexes.shape[0]):
        for j in range(0, 4):
            new_ind = indexes[i]
            dim = False
            break_0 = False

            while not break_0 and not dim:
                if new_ind[0] > x - 1 or new_ind[1] > y - 1 or new_ind[0] < 0 or new_ind[1] < 0:
                    dim = True
                elif player != board[new_ind[0], new_ind[1]]:
                    break_0 = True
                else:
                    sums[i, j] += 1
                    new_ind = new_ind + dirs[j]

            break_0 = dim = False
            sums[i, j] -= 1
            new_ind = indexes[i]

            while not break_0 and not dim:
                if new_ind[0] > x - 1 or new_ind[1] > y - 1 or new_ind[0] < 0 or new_ind[1] < 0:
                    dim = True
                elif player != board[new_ind[0], new_ind[1]]:
                    break_0 = True
                else:
                    sums[i, j] += 1
                    new_ind = new_ind - dirs[j]

    if np.sum(sums >= 4) > 0:
        return True
    else:
        return False


def check_end_state(board: np.ndarray, player: BoardPiece, last_action: PlayerAction = None) -> object:
    """ Checks the state of the game.

    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?

    Args:
        board: Current state of the board.
        player: Whose turn is it.
        last_action: Last action taken in the game.
    Returns:
        state_game: GameState.IS_WIN if player won, GameState. Otherwise,
                    GameState.STILL_PLAYING or GameState.IS_DRAW if board is full.
    """
    state_game = GameState.STILL_PLAYING

    if connected_four(board, player, last_action):
        state_game = GameState.IS_WIN
    elif np.sum(board == 0) == 0 and not connected_four(board, player, last_action):
        state_game = GameState.IS_DRAW

    return state_game
