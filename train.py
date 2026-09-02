import random
from collections import OrderedDict, deque
from pathlib import Path

import torch
import numpy as np

import config
import eval
import environment
from agent import Agent

print(torch.cuda.is_available())

env = environment.Connect4Env()
agent = Agent()
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
    if random.random() < 0.8:
        return agent
    return opponent_pool.sample()


file_train_id = random.randint(1000, 9999)
wins_per_50_ep = 0
total_steps = 0
for episode in range(500):
    state, info = env.reset()
    episode_step = 1
    episode_reward = 0
    agent_first = random.randint(0, 1)

    opponent = choose_opponent(agent)

    done = False
    while not done:

        if episode_step % 2 == agent_first:
            action = agent.select_action(state, env, info["action_mask"])
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.store_experience(state, action, reward, next_state, terminated, info["action_mask"])
            episode_reward += reward

        else:
            # 1. random opponent action
            # action = env.sample_action()

            # 2. past version opponent action
            action = opponent.select_action(state, env, info["action_mask"])

            next_state, reward, terminated, truncated, info = env.step(action)

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
        if wins_per_50_ep > 26:
            checkpoint_path = Path(config.CHECKPOINT_DIR) / f"ep_{episode + 1}_{file_train_id}.pth"
            agent.save(checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
            print()
        wins_per_50_ep = 0

    # print(f"Episode {episode + 1} - Steps: {episode_step}, Total Reward: {episode_reward}, ")
