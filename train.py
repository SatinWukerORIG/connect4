import random
from collections import deque
from pathlib import Path

import torch
import numpy as np

import config
import environment
from agent import Agent


env = environment.Connect4Env()
agent = Agent()

total_steps = 0
for episode in range(10):
    state, _ = env.reset()
    episode_step = 0
    total_reward = 0

    done = False
    while not done:

        if episode_step % 2 == 0:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.store_experience(state, action, reward, next_state, terminated)

        else:
            action = random.randint(0, 6)
            next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        if total_steps % config.TARGET_UPDATE_FREQ == 0:
            agent.update_network()

        if len(agent.memory_buffer) >= config.MIN_REPLAY_SIZE:
            batch = random.sample(agent.memory_buffer, config.BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.stack(states)
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.stack(next_states)
            dones = torch.tensor(dones, dtype=torch.float32)

            agent.train_step(states, actions, rewards, next_states, dones)

        state = next_state
        total_reward += reward
        episode_step += 1
        total_steps += 1

    print(f"Episode {episode + 1}/{config.EPISODES} - Steps: {episode_step}, Total Reward: {total_reward}")

