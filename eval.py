"""Benchmark the training agent against a pool of saved checkpoints.

Each game draws a random opponent from `eval_checkpoint/`, so the score covers a
mix of past versions instead of a single one. Call it from the training loop:

    wins = evaluate(agent)
    print(f"{wins}/50 vs the eval pool")
"""

import random
from pathlib import Path

import environment
from agent import Agent

EVAL_CHECKPOINT_DIR = Path("eval_checkpoint")

_opponents = {}  # path -> Agent, so checkpoints are read from disk only once


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


def evaluate(agent, games=50, opponent=None):
    """Play `games` greedy games against the eval pool; return the agent's wins.

    A fresh opponent is drawn from `eval_checkpoint/` for each game -- pass
    `opponent` to face one fixed player instead. The agent takes each seat for
    half the games, since moving first is an advantage. Draws and losses both
    count as "not a win".
    """
    paths = sorted(EVAL_CHECKPOINT_DIR.glob("*.pth"))

    env = environment.Connect4Env()
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0  # no exploration while measuring
    wins = 0

    try:
        for game in range(games):
            if opponent is not None:
                rival = opponent
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
