"""Benchmark the training agent against a pool of saved checkpoints.

Half the games are played against a fixed-strength minimax agent, the other half
against a random opponent drawn from `eval_checkpoint/`, so the score covers both
absolute strength and progress against past versions. Call it from the training
loop:

    wins = evaluate(agent)
    print(f"{wins}/50 vs the eval pool")

`evaluate_vs_checkpoints` and `evaluate_vs_minimax` split those two opponents
apart and report a win percentage each, so the training loop can gate saving on
both ("beats the past versions AND holds its own against search").
"""

import random
from pathlib import Path

import environment
import minimax
from agent import Agent

EVAL_CHECKPOINT_DIR = Path("eval_checkpoint")

MINIMAX_DEPTH = 3  # search depth of the minimax half of the pool

_opponents = {}  # path -> Agent, so checkpoints are read from disk only once
_minimax_opponent = minimax.MinimaxAgent(depth=MINIMAX_DEPTH)


class _RandomOpponent:
    """Stand-in for when the eval folder is empty."""

    def select_action(self, state, env, action_mask):
        return env.sample_action()


def _load_opponent(path):
    if path not in _opponents:
        opponent = Agent(inference_only=True)
        opponent.load(path)
        _opponents[path] = opponent
    return _opponents[path]


def _play_game(agent, rival, env, agent_player):
    """One greedy game; returns the winner (1/-1/0) in env-seat terms."""
    state, info = env.reset()

    while True:
        mover = agent if env.current_player == agent_player else rival
        action = mover.select_action(state, env, info["action_mask"])
        state, _reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            return info["winner"]


def _match(agent, rivals, env):
    """Play every rival twice -- once opening, once replying. Returns win %."""
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0  # no exploration while measuring
    wins = 0
    games = 0

    try:
        for rival in rivals:
            for agent_player in (1, -1):  # play both seats against each rival
                winner = _play_game(agent, rival, env, agent_player)
                wins += winner == agent_player
                games += 1
    finally:
        agent.epsilon = saved_epsilon

    return 100.0 * wins / games if games else 0.0


def evaluate_vs_checkpoints(agent):
    """Win % over two games against every checkpoint in `eval_checkpoint/`.

    Each opponent is faced twice, once from each seat, since moving first is an
    advantage -- 11 checkpoints means 22 games. Draws and losses both count as
    "not a win". Falls back to a single random opponent when the folder is empty.
    """
    paths = sorted(EVAL_CHECKPOINT_DIR.glob("*.pth"))
    rivals = [_load_opponent(path) for path in paths] or [_RandomOpponent()]

    return _match(agent, rivals, environment.Connect4Env())


def evaluate_vs_minimax(agent, games=20, depth=4):
    """Win % over `games` games against a depth-`depth` minimax agent.

    Half the games are played from each seat, so 20 games means 10 opening and
    10 replying.
    """
    rival = minimax.MinimaxAgent(depth=depth)

    return _match(agent, [rival] * (games // 2), environment.Connect4Env())


def evaluate(agent, games=50, opponent=None):
    """Play `games` greedy games against the eval pool; return the agent's wins.

    Half the games face the minimax agent; the other half face a fresh opponent
    drawn from `eval_checkpoint/` -- pass `opponent` to face one fixed player for
    every game instead. The agent takes each seat for half the games, since
    moving first is an advantage. Draws and losses both count as "not a win".
    """
    paths = sorted(EVAL_CHECKPOINT_DIR.glob("*.pth"))

    env = environment.Connect4Env()
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0  # no exploration while measuring
    wins = 0

    try:
        for game in range(games):
            # Alternate opponent type every two games, so each of them is faced
            # from both seats equally often.
            use_minimax = (game // 2) % 2 == 0

            if opponent is not None:
                rival = opponent
            elif use_minimax:
                rival = _minimax_opponent
            elif paths:
                rival = _load_opponent(random.choice(paths))
            else:
                rival = _RandomOpponent()

            agent_player = 1 if game % 2 == 0 else -1
            state, info = env.reset()

            while True:
                mover = agent if env.current_player == agent_player else rival
                action = mover.select_action(state, env, info["action_mask"])
                state, _reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    wins += info["winner"] == agent_player
                    break
    finally:
        agent.epsilon = saved_epsilon

    return wins


if __name__ == "__main__":
    import config

    agent = Agent()
    agent.load(f"{config.CHECKPOINT_DIR}/best_model.pth")
    print(f"{evaluate(agent)}/50 wins vs the eval pool")
    print(f"{evaluate_vs_checkpoints(agent):.1f}% vs the eval checkpoints")
    print(f"{evaluate_vs_minimax(agent):.1f}% vs minimax depth 4")
