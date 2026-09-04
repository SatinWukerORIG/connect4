import argparse
import itertools
import random
from collections import OrderedDict, deque
from pathlib import Path

import torch
import numpy as np

import config
import eval
import environment
from agent import Agent
from minimax import MinimaxAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Connect 4 DQN agent.")
    parser.add_argument(
        "--total-steps",
        type=int,
        default=config.TOTAL_STEPS,
        help="environment steps to train for; the run ends after the episode "
             "that crosses this count (default: %(default)s)",
    )
    parser.add_argument(
        "--train-mode",
        choices=("random", "selfplay", "minimax"),
        default=config.TRAIN_MODE,
        help="opponent to train against: 'random' plays uniformly random moves, "
             "'selfplay' plays the current net or a past checkpoint, 'minimax' is "
             "selfplay with 20%% of the opponent moves taken by minimax "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--initial-epsilon",
        type=float,
        default=config.EPSILON_START,
        help="exploration rate at step 0, decayed linearly to EPSILON_MIN over "
             "EPSILON_DECAY_STEPS (default: %(default)s)",
    )
    parser.add_argument(
        "--minimax-depth",
        type=int,
        default=4,
        help="search depth of the minimax opponent (default: %(default)s)",
    )
    return parser.parse_args()


args = parse_args()
train_mode = args.train_mode

print(f"training on {config.DEVICE}")
print(f"train_mode: {train_mode}, total_steps: {args.total_steps}, "
      f"initial epsilon: {args.initial_epsilon}")

env = environment.Connect4Env()
agent = Agent(epsilon_start=args.initial_epsilon)
minimax_agent = MinimaxAgent(depth=args.minimax_depth)
if Path(config.CHECKPOINT_DIR).exists():
    if Path(f'{config.CHECKPOINT_DIR}/best_model.pth').exists():
        agent.load(f'{config.CHECKPOINT_DIR}/best_model.pth')
        print(f"Loaded model from {config.CHECKPOINT_DIR}/best_model.pth")
else:
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

class RandomOpponent:
    """Fallback used before any checkpoint exists."""

    def select_action(self, state, env, action_mask):
        return env.sample_action()


class OpponentPool:
    """Keeps a few past-version opponents resident in memory.

    Building an Agent and reading its checkpoint off disk costs far more than
    the games it plays, so loaded opponents are cached and re-sampled instead
    of being rebuilt every episode. The checkpoint listing is re-globbed only
    every `refresh_every` draws so newly saved checkpoints still join the pool.
    """

    def __init__(self, checkpoint_dir, capacity, refresh_every):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.capacity = capacity
        self.refresh_every = refresh_every
        self.draws = 0
        self.paths = []
        self.loaded = OrderedDict()  # path -> Agent, least-recently-used first
        self.random_opponent = RandomOpponent()

    def sample(self):
        if self.draws % self.refresh_every == 0:
            self.paths = list(self.checkpoint_dir.glob("*.pth"))
            self.paths = random.sample(self.paths, k=min(len(self.paths), config.MAX_CHECKPOINTS))
        self.draws += 1

        if not self.paths:
            return self.random_opponent

        path = random.choice(self.paths)
        opponent = self.loaded.get(path)
        if opponent is None:
            opponent = Agent(inference_only=True)
            opponent.load(path)
            opponent.epsilon = 0.0  # disable exploration for the opponent
            self.loaded[path] = opponent
            if len(self.loaded) > self.capacity:
                self.loaded.popitem(last=False)  # evict least recently used
        else:
            self.loaded.move_to_end(path)
        return opponent


opponent_pool = OpponentPool(
    config.CHECKPOINT_DIR,
    capacity=config.OPPONENT_POOL_SIZE,
    refresh_every=config.OPPONENT_POOL_REFRESH_EVERY,
)


def choose_opponent(agent):
    # play against past version.
    # 80% current network, 20% a past checkpoint (random actions until one exists)
    if random.random() < 0.5:
        return agent
    return opponent_pool.sample()


file_train_id = random.randint(1000, 9999)
wins_per_50_ep = 0
total_steps = 0
for episode in itertools.count():
    if total_steps >= args.total_steps:
        break

    state, info = env.reset()
    episode_step = 1
    episode_reward = 0
    agent_first = random.randint(0, 1)

    opponent = None if train_mode == 'random' else choose_opponent(agent)

    done = False
    while not done:

        if episode_step % 2 == agent_first:
            action = agent.select_action(state, env, info["action_mask"])
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        else:
            if train_mode == 'random':
                # 1. random opponent action
                action = env.sample_action()
            elif train_mode == 'minimax' and random.random() < 0.2:
                # 2. minimax opponent action. Search is slow, so it only takes a
                # fifth of the moves; the rest still come from the pool.
                action = minimax_agent.select_action(state, env, info["action_mask"])
            else:
                # 3. current-net or past-version opponent action
                action = opponent.select_action(state, env, info["action_mask"])

            next_state, reward, terminated, truncated, info = env.step(action)

        agent.store_experience(state, action, reward, next_state, terminated, info["action_mask"])
        done = terminated or truncated

        if total_steps % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        if len(agent.memory_buffer) >= config.MIN_REPLAY_SIZE:
            batch = random.sample(agent.memory_buffer, config.BATCH_SIZE)
            states, actions, rewards, next_states, dones, next_action_masks = zip(*batch)

            states = torch.stack(states)
            actions = torch.as_tensor(actions, dtype=torch.int64)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            next_states = torch.stack(next_states)
            dones = torch.as_tensor(dones, dtype=torch.float32)
            next_action_masks = torch.stack(next_action_masks)

            agent.train_step(states, actions, rewards, next_states, dones, next_action_masks)

        agent.update_epsilon(total_steps)
        state = next_state
        episode_step += 1
        total_steps += 1


    if episode_reward > 0:
        wins_per_50_ep += 1
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1} - Wins in last 50 episodes: {wins_per_50_ep}")
        print(f"epsilon: {agent.epsilon:.4f}, total_steps: {total_steps}, memory_buffer size: {len(agent.memory_buffer)}")
        if train_mode == 'random':
            if wins_per_50_ep > 40:
                checkpoint_path = Path(config.CHECKPOINT_DIR) / f"ep_{episode + 1}_{file_train_id}.pth"
                agent.save(checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
                print()
        else:
            print(f"Opponent pool size: {len(opponent_pool.loaded)}")
            print('Wins per 50 episodes:', wins_per_50_ep)
            # Gate saving on both benchmarks: better than the past versions and
            # at least even against search. The minimax games are the slow half,
            # so only play them when the checkpoint pool has already been passed.
            pool_pct = eval.evaluate_vs_checkpoints(agent)
            minimax_pct = eval.evaluate_vs_minimax(agent) if pool_pct > 70 else 0.0
            print(f"vs eval checkpoints: {pool_pct:.1f}% · vs minimax depth 4: {minimax_pct:.1f}%")
            print()

            if pool_pct > 70 and minimax_pct >= 50:
                checkpoint_path = Path(config.CHECKPOINT_DIR) / f"ep_{episode + 1}_{file_train_id}_{train_mode}.pth"
                agent.save(checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
                print()
        wins_per_50_ep = 0

print(f"Training finished after {episode} episodes / {total_steps} steps.")

