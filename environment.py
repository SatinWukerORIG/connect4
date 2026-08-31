"""Connect 4 environment for self-play DQN training.

Follows the gymnasium interface (reset -> (obs, info), step -> (obs, reward,
terminated, truncated, info)) without depending on gymnasium itself.

Self-play convention: the board is always returned from the perspective of the
player whose turn it is (+1 = me, -1 = opponent, 0 = empty), so a single network
can play both seats. Rewards are given to the player that just moved.
"""

import numpy as np

ROWS = 6
COLS = 7

PLAYER = 1
OPPONENT = -1

# Rewards, from the point of view of the agent that just moved.
REWARD_WIN = 1.0
REWARD_DRAW = 0.0
REWARD_STEP = 0.0
REWARD_INVALID = -1.0

# The four directions a line can run in: right, down, down-right, down-left.
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class Discrete:
    """Minimal stand-in for gymnasium.spaces.Discrete."""

    def __init__(self, n, seed=None):
        self.n = int(n)
        self.shape = ()
        self.dtype = np.int64
        self._rng = np.random.default_rng(seed)

    def sample(self, mask=None):
        if mask is None:
            return int(self._rng.integers(self.n))
        choices = np.flatnonzero(np.asarray(mask, dtype=bool))
        if choices.size == 0:
            raise ValueError("no valid actions to sample from")
        return int(self._rng.choice(choices))

    def contains(self, x):
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def __repr__(self):
        return f"Discrete({self.n})"


class Box:
    """Minimal stand-in for gymnasium.spaces.Box."""

    def __init__(self, low, high, shape, dtype=np.float32, seed=None):
        self.low = low
        self.high = high
        self.shape = tuple(shape)
        self.dtype = dtype
        self._rng = np.random.default_rng(seed)

    def sample(self):
        return self._rng.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

    def contains(self, x):
        x = np.asarray(x)
        return x.shape == self.shape and bool((x >= self.low).all() and (x <= self.high).all())

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def __repr__(self):
        return f"Box({self.low}, {self.high}, {self.shape}, {np.dtype(self.dtype).name})"


class Connect4Env:
    """Two-player Connect 4 as a single-agent env driven by alternating turns.

    One `step` plays one move. The caller is responsible for alternating between
    the two policies (or letting one policy play itself); `current_player` says
    whose move it is, and the observation is already flipped for them.
    """

    metadata = {"render_modes": [], "players": 2}

    def __init__(self, rows=ROWS, cols=COLS, flatten=True, seed=None):
        self.rows = rows
        self.cols = cols
        self.flatten = flatten

        obs_shape = (rows * cols,) if flatten else (rows, cols)
        self.observation_space = Box(-1.0, 1.0, obs_shape, np.float32, seed=seed)
        self.action_space = Discrete(cols, seed=seed)

        self.state_dim = int(np.prod(obs_shape))
        self.action_dim = cols

        self.board = np.zeros((rows, cols), dtype=np.int8)
        self.current_player = PLAYER
        self.move_count = 0
        self.terminated = False
        self.winner = 0
        self.last_move = None

        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ core

    def reset(self, seed=None, options=None):
        """Start a new game. Returns (observation, info)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

        self.board[:] = 0
        self.current_player = PLAYER
        self.move_count = 0
        self.terminated = False
        self.winner = 0
        self.last_move = None

        return self._observation(), self._info()

    def step(self, action):
        """Drop a piece in column `action` for the current player.

        Returns (observation, reward, terminated, truncated, info). The reward
        belongs to the player that just moved; the returned observation is for
        the *next* player, so flip the sign of the reward when you store the
        opponent's transition.
        """
        if self.terminated:
            raise RuntimeError("step() called on a finished game; call reset() first")

        action = int(action)
        if not self.action_space.contains(action) or not self.is_valid_action(action):
            # Illegal move loses the game outright. Masking actions during
            # action selection means this should never fire in training.
            self.terminated = True
            self.winner = -self.current_player
            info = self._info()
            info["invalid_action"] = True
            return self._observation(), REWARD_INVALID, True, False, info

        row = self._drop(action, self.current_player)
        self.last_move = (row, action)
        self.move_count += 1
        mover = self.current_player

        if self._is_winning_move(row, action, mover):
            self.terminated = True
            self.winner = mover
            reward = REWARD_WIN
        elif self.move_count == self.rows * self.cols:
            self.terminated = True
            self.winner = 0
            reward = REWARD_DRAW
        else:
            self.current_player = -mover
            reward = REWARD_STEP

        return self._observation(), reward, self.terminated, False, self._info()

    def close(self):
        pass

    # ------------------------------------------------------------- accessors

    def is_valid_action(self, action):
        return 0 <= action < self.cols and self.board[0, action] == 0

    def valid_actions(self):
        return np.flatnonzero(self.board[0] == 0)

    def action_mask(self):
        """Boolean mask over columns; use it to mask Q-values before argmax."""
        return self.board[0] == 0

    def sample_action(self):
        """Uniformly random *legal* move, for epsilon-greedy exploration."""
        return self.action_space.sample(mask=self.action_mask())

    def clone(self):
        """Deep copy of the env, e.g. for lookahead or evaluation rollouts."""
        other = Connect4Env(self.rows, self.cols, self.flatten)
        other.board = self.board.copy()
        other.current_player = self.current_player
        other.move_count = self.move_count
        other.terminated = self.terminated
        other.winner = self.winner
        other.last_move = self.last_move
        return other

    # -------------------------------------------------------------- internal

    def _drop(self, col, player):
        """Place `player`'s piece in `col` and return the row it landed on."""
        column = self.board[:, col]
        row = int(np.flatnonzero(column == 0)[-1])
        self.board[row, col] = player
        return row

    def _is_winning_move(self, row, col, player):
        """Only lines through the piece just played can be new wins."""
        for dr, dc in _DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while (
                    0 <= r < self.rows
                    and 0 <= c < self.cols
                    and self.board[r, c] == player
                ):
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                return True
        return False

    def _observation(self):
        """Board as seen by `current_player`: +1 theirs, -1 the opponent's."""
        obs = (self.board * self.current_player).astype(np.float32)
        return obs.reshape(-1) if self.flatten else obs

    def _info(self):
        return {
            "current_player": self.current_player,
            "valid_actions": self.valid_actions(),
            "action_mask": self.action_mask(),
            "winner": self.winner,
            "move_count": self.move_count,
            "last_move": self.last_move,
        }


def make(render_mode=None, **kwargs):
    """Gymnasium-style factory. `render_mode` is accepted but not implemented."""
    if render_mode is not None:
        raise NotImplementedError("render modes are not implemented yet")
    return Connect4Env(**kwargs)
