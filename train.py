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
if Path(f'{config.CHECKPOINT_DIR}/best_model.pth').exists():
    agent.load(f'{config.CHECKPOINT_DIR}/best_model.pth')

wins_per_50_ep = 0

total_steps = 0
for episode in range(500):
    state, info = env.reset()
    episode_step = 1
    episode_reward = 0
    agent_first = random.randint(0, 1)

    done = False
    while not done:

        if episode_step % 2 == agent_first:
            action = agent.select_action(state, env, info["action_mask"])
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.store_experience(state, action, reward, next_state, terminated)
            episode_reward += reward

        else:
            action = random.randint(0, 6)
            next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        if total_steps % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        if len(agent.memory_buffer) >= config.MIN_REPLAY_SIZE:
            batch = random.sample(agent.memory_buffer, config.BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.stack(states)
            actions = torch.as_tensor(actions, dtype=torch.int64)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            next_states = torch.stack(next_states)
            dones = torch.as_tensor(dones, dtype=torch.float32)

            agent.train_step(states, actions, rewards, next_states, dones)

        state = next_state
        episode_step += 1
        total_steps += 1


    if episode_reward > 0:
        wins_per_50_ep += 1
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1} - Wins in last 50 episodes: {wins_per_50_ep}")
        if wins_per_50_ep > 25:
            checkpoint_path = Path(config.CHECKPOINT_DIR) / f"ep_{episode + 1}.pth"
            torch.save(agent.online_network.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
        wins_per_50_ep = 0

    print(f"Episode {episode + 1} - Steps: {episode_step}, Total Reward: {episode_reward}, ")
