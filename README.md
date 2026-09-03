# Connect 4 DQN

A Connect 4 agent trained with Double DQN self-play, plus a Tkinter window to play against it.

## Requirements

Python with `torch`, `numpy`, and `tkinter`. On this machine torch lives in miniforge, so use
`/Users/sherlockli/miniforge3/bin/python3` — the system `/usr/bin/python3` has numpy but no torch.

## Files

| File | What it does |
| --- | --- |
| `environment.py` | The game. Gymnasium-style `reset()` / `step()`, no gymnasium dependency. The observation is a `(2, 6, 7)` plane stack — `[my pieces, opponent pieces]` — always flipped to whoever's turn it is, so one network plays both seats. Rewards go to the player that just moved: `+1` win, `0` draw/step, `-1` illegal move. |
| `model.py` | `Connect4Model`: a 2→64 conv stem, 3 residual blocks, then two linear layers to 7 Q-values (one per column). |
| `agent.py` | `Agent`: online + target networks, replay buffer, epsilon-greedy action selection, Double DQN `train_step`, and `save` / `load`. |
| `config.py` | Hyperparameters — batch size, replay sizes, `GAMMA`, epsilon decay, learning rate, checkpoint dir. |
| `train.py` | The training loop (see below). |
| `play.py` | Play against a checkpoint in a window (see below). |
| `eval.py` | Stub — loads `checkpoint/best_model.pth` and does nothing else yet. |

## Training

```
/Users/sherlockli/miniforge3/bin/python3 train.py
```

Runs 2000 episodes against a random opponent, alternating who moves first. Every 50 episodes it prints
the win count and, if the agent won more than 40 of those 50, saves a checkpoint to
`checkpoint/ep_<episode>_<runid>.pth`. `<runid>` is random per run, so separate runs don't overwrite
each other. If `checkpoint/best_model.pth` exists it is loaded first and training resumes from it.

A checkpoint is a dict with `model`, `target_model`, `optimizer`, and `epsilon`, so training can pick up
where it stopped.

Knobs live in `config.py`. The episode count is hardcoded at `train.py:89`.

`train.py` also contains an opponent pool that samples past checkpoints to play against instead of a
random opponent — it's wired up but currently commented out at lines 95 and 110.

## Playing

```
/Users/sherlockli/miniforge3/bin/python3 play.py                                              # vs random
/Users/sherlockli/miniforge3/bin/python3 play.py --agent dqn --checkpoint checkpoint/ep_1550_1676.pth
```

Click a column (or press `1`–`7`) to drop a disc. `N` starts a new game, `Q` quits. Buttons let you
start over or swap who moves first; `--second` lets the agent open.

Other flags: `--device` (default `cpu`), `--seed`.

Any object with an `act(obs, action_mask) -> int` method works as an opponent, so a new policy only
needs that method plus a line in `build_agent`.
