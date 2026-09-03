"""Benchmark the training agent against the saved best model.

Call `evaluate(agent)` from the training loop to see how the agent is doing
against `checkpoint/best_model.pth`:

    wins = evaluate(agent)
    print(f"{wins}/50 vs best_model")
"""

from pathlib import Path

import config
import environment
from agent import Agent

BEST_MODEL = Path(config.CHECKPOINT_DIR) / "best_model.pth"

_opponent = None  # loaded once, not re-read on every call


class _RandomOpponent:
    """Stand-in for before best_model.pth exists."""

    def select_action(self, state, env, action_mask):
        return env.sample_action()


def _best_opponent():
    global _opponent
    if _opponent is None:
        if BEST_MODEL.exists():
            _opponent = Agent(inference_only=True)
            _opponent.load(BEST_MODEL)
        else:
            _opponent = _RandomOpponent()
    return _opponent


def evaluate(agent, games=50, opponent=None):
    """Play `games` greedy games against best_model.pth; return the agent's wins.

    The agent takes each seat for half the games, since moving first is an
    advantage. Draws and losses both count as "not a win".
    """
    if opponent is None:
        opponent = _best_opponent()

    env = environment.Connect4Env()
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0  # no exploration while measuring
    wins = 0

    try:
        for game in range(games):
            agent_player = 1 if game % 2 == 0 else -1
            state, info = env.reset()

            while True:
                mover = agent if env.current_player == agent_player else opponent
                action = mover.select_action(state, env, info["action_mask"])
                state, _reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    wins += info["winner"] == agent_player
                    break
    finally:
        agent.epsilon = saved_epsilon

    return wins


if __name__ == "__main__":
    agent = Agent()
    agent.load(BEST_MODEL)
    print(f"{evaluate(agent)}/50 wins vs best_model")
