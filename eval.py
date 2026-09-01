"""Evaluate a checkpoint against a random agent, then compare it to a baseline.

    python eval.py --checkpoint checkpoint/dqn.pt
    python eval.py --checkpoint checkpoint/dqn.pt --baseline checkpoint/old.pt

Each model plays `--games` games (default 50) against a random opponent,
alternating who moves first so neither side gets the first-move advantage. Both
models face the same seeded opponent, so the win rates are comparable.

With no --baseline, the baseline is an untrained (randomly initialised)
Connect4Model -- the "did training do anything at all?" reference point.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from environment import Connect4Env, PLAYER
from model import Connect4Model


def load_model(path, device):
    """Load a checkpoint saved as a bare state_dict or {"model_state_dict": ...}."""
    blob = torch.load(path, map_location=device)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        blob = blob["model_state_dict"]
    net = Connect4Model().to(device)
    net.load_state_dict(blob)
    net.eval()
    return net


def untrained_model(device, seed=0):
    torch.manual_seed(seed)
    net = Connect4Model().to(device)
    net.eval()
    return net


def greedy_action(net, obs, mask, device):
    """Best legal column. Also reports whether the raw argmax was already legal."""
    with torch.no_grad():
        state = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q = net(state).squeeze(0)
    was_legal = bool(mask[int(torch.argmax(q).item())])
    mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=device)
    return int(torch.argmax(q.masked_fill(~mask_t, -math.inf)).item()), was_legal


def play_game(net, env, rng, model_first, device):
    """One game vs. a random (legal-move) opponent. Returns (result, illegal_count).

    result is +1 win / 0 draw / -1 loss, from the model's point of view.
    """
    obs, _info = env.reset()
    model_seat = PLAYER if model_first else -PLAYER
    illegal = 0

    while True:
        # `obs` is always from the perspective of whoever is about to move.
        mask = env.action_mask()
        if env.current_player == model_seat:
            action, was_legal = greedy_action(net, obs, mask, device)
            illegal += not was_legal
        else:
            action = int(rng.choice(np.flatnonzero(mask)))

        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            winner = info["winner"]
            if winner == 0:
                return 0, illegal
            return (1 if winner == model_seat else -1), illegal


def evaluate(net, games, seed, device):
    """Play `games` games against a random agent, alternating who opens."""
    env = Connect4Env()
    rng = np.random.default_rng(seed)
    wins = losses = draws = illegal = 0

    for i in range(games):
        result, game_illegal = play_game(net, env, rng, model_first=(i % 2 == 0), device=device)
        illegal += game_illegal
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            draws += 1

    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games,
        "score": (wins + 0.5 * draws) / games,  # win rate with draws counted as half
        "illegal": illegal,
    }


def stderr(rate, games):
    """Standard error of a win rate over `games` Bernoulli trials."""
    return math.sqrt(max(rate * (1 - rate), 0.0) / games)


def report(name, r):
    err = stderr(r["win_rate"], r["games"])
    print(
        f"{name:<14} {r['wins']:>3}W {r['losses']:>3}L {r['draws']:>3}D   "
        f"win rate {r['win_rate']:6.1%} +/- {err:.1%}   score {r['score']:6.1%}   "
        f"illegal picks {r['illegal']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="checkpoint to evaluate")
    parser.add_argument("--baseline", default=None,
                        help="checkpoint to compare against (default: untrained network)")
    parser.add_argument("--games", type=int, default=50, help="games per model (default: 50)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the random opponent")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    path = Path(args.checkpoint)
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}")
    model = load_model(path, device)

    if args.baseline is None:
        baseline_name, baseline = "untrained", untrained_model(device)
    else:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            raise SystemExit(f"baseline not found: {baseline_path}")
        baseline_name, baseline = baseline_path.stem, load_model(baseline_path, device)

    print(f"{args.games} games each vs. a random agent (seed {args.seed}), sides alternating\n")
    model_result = evaluate(model, args.games, args.seed, device)
    baseline_result = evaluate(baseline, args.games, args.seed, device)

    report(path.stem, model_result)
    report(baseline_name, baseline_result)

    delta = model_result["win_rate"] - baseline_result["win_rate"]
    verdict = "better than" if delta > 0 else "worse than" if delta < 0 else "level with"
    print(f"\n{path.stem} is {verdict} {baseline_name}: {delta:+.1%} win rate")
    if abs(delta) < 2 * stderr(model_result["win_rate"], args.games):
        print(f"(within noise at {args.games} games -- raise --games to tell them apart)")


if __name__ == "__main__":
    main()
