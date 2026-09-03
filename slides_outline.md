# Connect 4 DQN — Presentation Outline

Content for ~12 slides. Each slide has a title, the bullets to show, and a short
"say this" note that does not need to go on the slide.

---

## 1. Title

**Learning Connect 4 from Scratch — a Double DQN with Self-Play**

- Deep reinforcement learning agent for Connect 4
- No human games, no opening book, no hand-written strategy
- Trained purely from experience: random play → self-play → search-based opponent

*Say:* The agent starts knowing only the rules. Everything it learns about
threats, blocking and center control emerges from reward.

---

## 2. The Problem

- Board: 6 rows × 7 columns, 7 possible moves per turn
- Two players, perfect information, zero-sum, alternating turns
- ~4.5 × 10¹² legal positions — too many to tabulate, so the value function
  must be *approximated*
- Goal: a policy that picks a column given a board

*Say:* This is why we need a neural network rather than a lookup table, and why
we need a convolutional one rather than a flat MLP.

---

## 3. State Representation — the (2, 6, 7) Tensor

- The board is encoded as **two binary planes** stacked into shape `(2, 6, 7)`
  - Plane 0 = the current mover's pieces (1 where present, 0 elsewhere)
  - Plane 1 = the opponent's pieces
- **Always from the mover's point of view**, so the same network plays both seats
- Why not one plane of {-1, 0, +1}? Two planes keep "empty" and "opponent"
  linearly separable and let the first conv layer weight each side independently
- Why planes at all? It's an image — a 6×7 image with 2 channels — so a CNN
  applies directly

*Say:* This one design choice is what makes self-play possible: there is no
"player 1 network" and "player 2 network," only one network that always sees
the board as *mine vs. theirs*.

*Visual:* show a small board and its two 0/1 grids side by side.

---

## 4. Network Architecture — CNN + Residual Blocks

```
input (2, 6, 7)
  └─ Conv2d(2 → 64, 3×3, pad 1) + ReLU        # stem
  └─ ResidualBlock(64)                         # ×3
  └─ Flatten  → 64 × 6 × 7 = 2688
  └─ Linear(2688 → 128) + ReLU
  └─ Linear(128 → 7)                           # one Q-value per column
```

- **Why convolutions?** A win is a *local* pattern — four in a line. A 3×3 kernel
  detects it anywhere on the board, and the same detector is reused at every
  position (translation invariance + far fewer parameters than an MLP)
- **Why residual blocks?** `out = ReLU(conv2(ReLU(conv1(x))) + x)` — the skip
  connection lets gradients reach the early layers, so depth adds pattern
  complexity without the network becoming untrainable
- **Padding = 1, no pooling:** the 6×7 resolution is preserved end to end. The
  board is already tiny; downsampling would destroy exactly the spatial detail
  that decides the game
- Output = 7 Q-values, one per column: "expected return if I drop here"
- ≈ 0.57M parameters

*Say:* Three residual blocks give a receptive field wide enough to see a
four-in-a-row threat and the square that blocks it.

---

## 5. Learning Algorithm — Double DQN

- **Online network** selects; **target network** evaluates → removes the
  overestimation bias of vanilla DQN
- **Replay buffer** (200k transitions) breaks the correlation between
  consecutive moves; batches of 64 sampled uniformly
- **Target network** synced from the online network every 500 steps for a
  stationary regression target
- **Action masking:** full columns get their Q-value set to `-inf` before the
  argmax, so an illegal move can never be chosen
- **ε-greedy exploration:** ε decays linearly 1.0 → 0.01 over 20,000 steps
- Loss: Smooth L1 (Huber) · Optimizer: Adam, lr 1e-4 · γ = 0.99 · gradient
  norm clipped at 1.0

---

## 6. The Sign Flip — a Two-Player Bellman Equation

Standard DQN:
```
target = r + γ · max Q(s′, a′)
```

Ours:
```
target = r − γ · max Q(s′, a′)
```

- In a two-player game, `s′` is the board **as the opponent sees it**
- A position that is *good for them* is *bad for me* — so the bootstrapped value
  enters with a **minus sign**
- This is negamax logic expressed inside the Bellman update

*Say:* This was one of the real bugs we hit — with a plus sign the agent
happily learned to hand the opponent a win. Worth a slide because it is the
single most important consequence of the shared-perspective state design.

*Also note:* the mask is applied to the online network's next-state Q-values but
**not** the target network's — masking both would give `-inf × 0 = NaN` on a
full board.

---

## 7. Reward Design

| Event | Reward |
| --- | --- |
| Win | +1.0 |
| Draw | 0.0 |
| Non-terminal move | 0.0 |
| Illegal move | −1.0 |

- Deliberately **sparse** — no shaping, no "reward for blocking," no "reward
  for center column"
- The agent has to discover *why* a move was good by propagating the terminal
  reward backwards through γ
- Cleaner credit assignment: shaping risks teaching the proxy instead of the game

---

## 8. Data Augmentation — Free Symmetry

- Connect 4 is **mirror-symmetric left-to-right**
- Every stored transition is stored **twice**: the original, and the horizontally
  flipped board with the column index flipped `col → 6 − col`
- Same reward, same validity — it is a genuinely equivalent position
- Effect: **2× the training data at zero extra game-playing cost**, and the
  network is nudged toward learning the symmetry rather than memorizing one side

---

## 9. Training — Stage 1: Random Opponent (2000 episodes)

- Opponent plays uniformly random legal moves
- Agent's seat alternates each episode (moving first is an advantage — it must
  learn both)
- Every 50 episodes: print the win count; **save a checkpoint if wins > 40/50**
- Purpose: bootstrap the basics — complete your own line, block theirs, don't
  waste moves

*Say:* Random is a weak but *essential* opponent. It gives dense, varied
experience and gets the replay buffer full of legal, meaningful transitions
before we make the problem harder. The best of these checkpoints is promoted to
`best_model.pth` and becomes the seed for stage 2.

---

## 10. Training — Stage 2: Self-Play (2000 episodes)

- **80% of episodes:** the agent plays against **its own current network**
- **20% of episodes:** against a **randomly drawn past checkpoint** from a pool

**Why the 80/20 split?**

- Pure self-play collapses — the agent overfits to its own latest quirks and
  can cycle (learns to beat itself, forgets what beat it before)
- The 20% past-checkpoint slice is a defense against that **strategy
  forgetting**: it must stay strong against everything it used to be
- An `OpponentPool` keeps up to 8 past agents resident in memory and re-scans the
  checkpoint directory every 10 draws, so newly saved versions join the pool

*Say:* This is the classic self-play curriculum — the opponent gets stronger at
exactly the rate the agent does, so the difficulty is always right at the edge
of its ability.

---

## 11. Training — Stage 3: Minimax Opponent

- Opponent = **negamax search with alpha-beta pruning, depth 4**
- Heuristic for non-terminal leaves:
  - center-column control weighted ×3 (central squares belong to more lines)
  - every 4-cell window scored: 3-in-a-row = +5, 2-in-a-row = +2, opponent
    3-in-a-row = −4, blocked windows = 0
  - ties broken randomly so games don't repeat identically
- Purpose: a **fixed, principled, non-exploitable** benchmark opponent

*Say:* Self-play tells you the agent is better than *itself*. Minimax tells you
it is good at *Connect 4*. Training against it forces the agent to handle
tactically correct play — an opponent that never blunders and always takes an
available win.

---

## 12. Evaluation Strategy

- Run every 50 episodes, **50 games**, exploration off (ε = 0, greedy play)
- **50% vs. minimax (depth 3)** — absolute strength against a fixed standard
- **50% vs. a random checkpoint from the eval pool** — relative progress against
  past selves
- Opponent type alternates every 2 games and the agent **alternates seats**, so
  each opponent is faced equally often from both sides
- Draws count as non-wins — a deliberately strict score

**Checkpoint promotion gate** — a model is only saved if it clears *both*:
- `eval_score ≥ 30 / 50` (evaluation strength), **and**
- `wins ≥ 29 / 50` (training-time performance)

*Say:* Two independent halves matter. Minimax alone could be gamed by learning
one anti-search line; past checkpoints alone can drift upward while the agent
gets *worse* in absolute terms. Requiring both keeps a checkpoint honest.

---

## 13. Engineering Notes (optional slide)

- Gymnasium-style `reset()` / `step()` interface, no gymnasium dependency
- Device auto-select: CUDA → Apple MPS → CPU; replay buffer stays on CPU and
  only the sampled batch moves to GPU
- Checkpoints store `model`, `target_model`, `optimizer` and `epsilon`, so a run
  resumes exactly where it stopped
- Opponents load as `inference_only` agents — no target network, no optimizer,
  no replay buffer
- Every opponent (DQN, minimax, random) implements the same
  `select_action(state, env, action_mask)` interface, so they are
  interchangeable in training, evaluation and the play GUI
- `play.py` — Tkinter GUI to play the trained agent yourself

---

## 14. Results / Demo

*(fill in your final numbers)*

- Best checkpoint: ____ / 50 vs. the mixed evaluation pool
- vs. minimax depth 3: ____
- vs. random: ____
- **Live demo:** `python play.py --agent dqn --checkpoint checkpoint/best_model.pth`

---

## 15. What We Learned / Future Work

**Learned**

- State representation drives everything — the shared-perspective 2-plane
  encoding is what made one network able to play both seats
- The Bellman sign flip is the non-obvious core of two-player value learning
- Curriculum matters more than raw episode count: random → self-play → minimax
- Evaluation must be multi-faceted; a single opponent gives a misleading score

**Future work**

- Prioritized experience replay
- MCTS + policy/value head (AlphaZero-style) instead of pure Q-learning
- Deeper minimax opponents as a curriculum ladder
- Dueling DQN architecture (separate value and advantage streams)

---

### Notes for you before you build the deck

- `train.py:94` currently reads `range(1000)`; the 2000-episode runs described
  above were the actual experiments. Either bump it back to 2000 or say "up to
  2000 episodes per stage" on the slide.
- Stage selection is the `train_mode` flag at `train.py:16`
  (`"random"` / `"selfplay"` / `"minimax"`) — worth showing if you want to
  demonstrate how the three stages are run.
- Best visuals to make: (1) the two-plane state diagram, (2) the network block
  diagram, (3) a three-box curriculum arrow, (4) the eval pie split 50/50.
