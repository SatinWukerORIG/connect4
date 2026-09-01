import random
from collections import deque

import torch

import config
import model

class Agent:
    def __init__(self):
        self.online_network = model.Connect4Model()
        self.target_network = model.Connect4Model()
        self.target_network.load_state_dict(self.online_network.state_dict())

        self.memory_buffer = deque(maxlen=50_000)

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=config.LR)
        self.epsilon = 1.0

    def select_action(self, state, env, action_mask):
        if random.random() < self.epsilon:
            action = env.sample_action()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.online_network(state_tensor).squeeze(0)  # (7,)
                q_values = q_values.masked_fill(~action_mask, float('-inf'))
                action = torch.argmax(q_values).item()
        return action

    def train_step(self, states, actions, rewards, next_states, dones):
        # Step 1: compute predicted Q-values for the current states and actions
        predicted_q = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Step 2: compute target Q-values for the next states
        with torch.no_grad():
            best_actions = self.online_network(next_states).argmax(dim=1, keepdim=True)
            best_next_q = self.target_network(next_states).gather(1, best_actions).squeeze(1)
            target_q = rewards + config.GAMMA * best_next_q * (1 - dones)

        # Step 3: compute the loss and backprop using predicted and target Q-values
        loss = torch.nn.functional.smooth_l1_loss(predicted_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_epsilon()

    def store_experience(self, state, action, reward, next_state, done):
        # Convert to tensors
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        action_tensor = torch.as_tensor(action, dtype=torch.int64)
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32)
        done_tensor = torch.as_tensor(done, dtype=torch.float32)

        # Store in memory buffer
        self.memory_buffer.append((state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor))

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * config.EPSILON_DECAY, config.EPSILON_MIN)

    def save(self, path):
        torch.save({
            "model": self.online_network.state_dict(),
            "target_model": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu")

        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
                state_dict = checkpoint
            else:
                raise ValueError(f"Unsupported checkpoint format in {path}")

            self.online_network.load_state_dict(state_dict)

            if isinstance(checkpoint, dict) and "target_model" in checkpoint:
                self.target_network.load_state_dict(checkpoint["target_model"])
            else:
                self.update_target_network()

            if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            if isinstance(checkpoint, dict) and "epsilon" in checkpoint:
                self.epsilon = checkpoint["epsilon"]
        else:
            self.online_network.load_state_dict(checkpoint)
            self.update_target_network()