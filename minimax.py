"""Minimax agent for Connect 4, usable anywhere an Agent is.

It exposes the same `select_action(state, env, action_mask)` signature the DQN
agent uses, so it can be dropped straight into `eval.py` or `play.py`:

    import minimax
    action = minimax.select_action(state, env, info["action_mask"])

The search reads the board out of `env` and runs negamax with alpha-beta
pruning to a fixed depth, scoring unfinished positions with a simple count of
open three/two-in-a-rows.
"""

import random

import numpy as np

DEPTH = 4  # plies of lookahead; 4-6 is a reasonable range

WIN_SCORE = 10_000.0

# Same four line directions the environment checks: right, down, down-right,
# down-left.
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def select_action(state, env, action_mask=None, depth=DEPTH):
    """Pick a column for `env.current_player` by searching `depth` plies.

    `state` is accepted (and ignored) so this matches the Agent interface --
    the search needs the raw board, not the two-plane observation.
    """
    board = np.array(env.board, dtype=np.int8)
    player = env.current_player

    moves = _valid_columns(board)
    if len(moves) == 0:
        raise ValueError("no legal moves; the game is already over")

    best_score = -np.inf
    best_moves = []

    for col in moves:
        row = _drop(board, col, player)
        if _is_winning_move(board, row, col, player):
            score = WIN_SCORE
        else:
            score = -_negamax(board, -player, depth - 1, -np.inf, np.inf)
        board[row, col] = 0

        if score > best_score:
            best_score = score
            best_moves = [col]
        elif score == best_score:
            best_moves.append(col)

    # Break ties randomly so repeated games don't play out identically.
    return int(random.choice(best_moves))


# ---------------------------------------------------------------- the search


def _negamax(board, player, depth, alpha, beta):
    """Score the position for `player`, who is to move. Higher is better."""
    moves = _valid_columns(board)
    if len(moves) == 0:
        return 0.0  # board full: a draw
    if depth == 0:
        return _evaluate(board, player)

    best = -np.inf
    for col in moves:
        row = _drop(board, col, player)
        if _is_winning_move(board, row, col, player):
            # Prefer faster wins (and slower losses) by fading the score with depth.
            score = WIN_SCORE + depth
        else:
            score = -_negamax(board, -player, depth - 1, -beta, -alpha)
        board[row, col] = 0

        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break  # the opponent would never allow this line

    return best


# ------------------------------------------------------------ board helpers


def _valid_columns(board):
    return np.flatnonzero(board[0] == 0)


def _drop(board, col, player):
    """Place `player`'s piece in `col` in place and return the row it landed on."""
    row = int(np.flatnonzero(board[:, col] == 0)[-1])
    board[row, col] = player
    return row


def _is_winning_move(board, row, col, player):
    """Only lines through the piece just played can be new wins."""
    rows, cols = board.shape
    for dr, dc in _DIRECTIONS:
        count = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < rows and 0 <= c < cols and board[r, c] == player:
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= 4:
            return True
    return False


def _evaluate(board, player):
    """Heuristic score of a quiet position from `player`'s point of view."""
    rows, cols = board.shape
    score = 0.0

    # Center control: central columns take part in more winning lines.
    center = board[:, cols // 2]
    score += 3.0 * (int((center == player).sum()) - int((center == -player).sum()))

    # Every window of four cells contributes according to how close it is to a win.
    for r in range(rows):
        for c in range(cols):
            for dr, dc in _DIRECTIONS:
                end_r, end_c = r + 3 * dr, c + 3 * dc
                if not (0 <= end_r < rows and 0 <= end_c < cols):
                    continue
                window = [board[r + i * dr, c + i * dc] for i in range(4)]
                score += _window_score(window, player)

    return score


def _window_score(window, player):
    mine = window.count(player)
    theirs = window.count(-player)
    if mine and theirs:
        return 0.0  # blocked: neither side can complete this line
    if mine == 3:
        return 5.0
    if mine == 2:
        return 2.0
    if theirs == 3:
        return -4.0  # slightly less than a win of our own, so threats get met
    if theirs == 2:
        return -2.0
    return 0.0


class MinimaxAgent:
    """Object wrapper, for callers that expect an agent instance."""

    def __init__(self, depth=DEPTH):
        self.depth = depth
        self.epsilon = 0.0  # eval.py saves/restores this on whatever it is given

    def select_action(self, state, env, action_mask=None):
        return select_action(state, env, action_mask, depth=self.depth)
