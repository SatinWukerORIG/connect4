import random
from collections import deque

import torch

import config
import model

class Agent:
    def __init__(self, inference_only=False, device=None):
        self.inference_only = inference_only
        self.device = torch.device(device or config.DEVICE)
        self.online_network = model.Connect4Model().to(self.device)

        if inference_only:
            # Frozen opponent: it never learns, so skip the target network,
            # the optimizer and the replay buffer.
            self.online_network.eval()
            self.target_network = None
            self.optimizer = None
            self.memory_buffer = None
            self.epsilon = 0.0
            return

        self.target_network = model.Connect4Model().to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())

        self.memory_buffer = deque(maxlen=config.MAX_REPLAY_SIZE)

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=config.LR)
        self.epsilon = 1.0

    def select_action(self, state, env, action_mask):
        if random.random() < self.epsilon:
            action = env.sample_action()
        else:
            with torch.no_grad():
                state_tensor = torch.as_tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                q_values = self.online_network(state_tensor).squeeze(0)  # (7,)
                mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
                q_values = q_values.masked_fill(~mask_tensor, float('-inf'))
                action = torch.argmax(q_values).item()
        return action

    def train_step(self, states, actions, rewards, next_states, dones, next_action_masks):
        # The replay buffer lives on the CPU; only the sampled batch goes to the GPU.
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_action_masks = next_action_masks.to(self.device)

        # Step 1: compute predicted Q-values for the current states and actions
        predicted_q = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Step 2: compute target Q-values for the next states. Only the online
        # network's Q-values are masked -- masking the target network too would
        # make a full board (every column illegal) produce -inf * 0 = NaN.
        with torch.no_grad():
            next_q = self.online_network(next_states)
            next_q = next_q.masked_fill(~next_action_masks, float('-inf'))
            best_actions = next_q.argmax(dim=1, keepdim=True)
            best_next_q = self.target_network(next_states).gather(1, best_actions).squeeze(1)
            target_q = rewards - config.GAMMA * best_next_q * (1 - dones)

        # Step 3: compute the loss and backprop using predicted and target Q-values
        loss = torch.nn.functional.smooth_l1_loss(predicted_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

    def store_experience(self, state, action, reward, next_state, done, next_action_mask):
        # Convert to tensors
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        action_tensor = torch.as_tensor(action, dtype=torch.int64)
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32)
        done_tensor = torch.as_tensor(done, dtype=torch.float32)
        next_mask_tensor = torch.as_tensor(next_action_mask, dtype=torch.bool)

        # Store in memory buffer
        self.memory_buffer.append((state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor, next_mask_tensor))

        # Connect 4 is symmetric left-to-right, so the mirrored board with the
        # mirrored column played is an equally valid transition with the same
        # reward. Storing it too doubles the data for free.
        cols = state_tensor.shape[-1]
        self.memory_buffer.append((
            torch.flip(state_tensor, dims=[-1]),
            torch.as_tensor(cols - 1 - action, dtype=torch.int64),
            reward_tensor,
            torch.flip(next_state_tensor, dims=[-1]),
            done_tensor,
            torch.flip(next_mask_tensor, dims=[-1]),
        ))

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def update_epsilon(self, total_steps):
        frac = min(1.0, total_steps / config.EPSILON_DECAY_STEPS)
        self.epsilon = 0.6 + frac * (config.EPSILON_MIN - 0.6)

    def save(self, path):
        torch.save({
            "model": self.online_network.state_dict(),
            "target_model": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

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

            if self.inference_only:
                return

            if "target_model" in checkpoint:
                self.target_network.load_state_dict(checkpoint["target_model"])
            else:
                self.update_target_network()

            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            if "epsilon" in checkpoint:
                self.epsilon = checkpoint["epsilon"]
        else:
            self.online_network.load_state_dict(checkpoint)
            if not self.inference_only:
                self.update_target_network()
