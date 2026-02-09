"""
DRL Agent implementations:
- DQN Agent (with experience replay)
- Dueling DQN Agent (with experience replay)
- PPO Agent (with transition buffer)

Each agent implements:
- select_action: epsilon-greedy (DQN/Dueling) or policy sampling (PPO)
- learn: update networks from buffered experiences
- train_episode: run one full episode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional
import copy
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEVICE, NUM_ACTIONS, DISCOUNT_FACTOR, BATCH_SIZE, NUM_EPISODES,
    LEARNING_RATE, REPLAY_BUFFER_SIZE, SYNC_TARGET_STEPS,
    EPSILON_START, EPSILON_END, EPSILON_DECAY_DIVISOR,
    ILLEGAL_ACTION_VALUE,
    PPO_ACTOR_LR, PPO_CRITIC_LR, PPO_CLIP_EPSILON,
    PPO_GAE_LAMBDA, PPO_EPOCHS, PPO_CRITIC_DISCOUNT, PPO_ENTROPY_BETA,
    RESOLUTIONS, NUM_CANDLES
)
from networks import DQNNetwork, DuelingDQNNetwork, PPOActorNetwork, PPOCriticNetwork


# ============================================================
# Experience / Transition containers
# ============================================================

DQNExperience = namedtuple('DQNExperience', ['state', 'action', 'reward', 'next_state', 'done', 'mask', 'next_mask'])
PPOTransition = namedtuple('PPOTransition', ['state', 'action', 'log_prob', 'value', 'reward', 'done', 'mask'])


def obs_to_device(obs: dict, device: torch.device) -> dict:
    """Move observation dict tensors to device."""
    result = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            result[k] = torch.tensor(v, dtype=torch.float32).to(device)
        elif isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        else:
            result[k] = v
    return result


def batch_obs(obs_list: list, device: torch.device) -> dict:
    """Batch a list of observation dicts into a single dict of batched tensors."""
    keys = obs_list[0].keys()
    batched = {}
    for k in keys:
        vals = [o[k] for o in obs_list]
        if isinstance(vals[0], np.ndarray):
            batched[k] = torch.tensor(np.stack(vals), dtype=torch.float32).to(device)
        elif isinstance(vals[0], torch.Tensor):
            batched[k] = torch.stack(vals).to(device)
    return batched


# ============================================================
# Replay Buffer for DQN
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: DQNExperience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[DQNExperience]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

    @property
    def is_full(self):
        return len(self.buffer) >= self.buffer.maxlen


# ============================================================
# DQN Agent
# ============================================================

class DQNAgent:
    def __init__(self, use_images: bool = True, raw_dim: int = 336):
        self.use_images = use_images
        self.behavior_net = DQNNetwork(use_images, raw_dim).to(DEVICE)
        self.target_net = DQNNetwork(use_images, raw_dim).to(DEVICE)
        self.target_net.load_state_dict(self.behavior_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.behavior_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer()
        self.obs_index = 0
        self.total_steps = 0

    def get_epsilon(self) -> float:
        return max(EPSILON_END, EPSILON_START - self.obs_index / EPSILON_DECAY_DIVISOR)

    def select_action(self, obs: dict, action_mask: np.ndarray,
                      explore: bool = True) -> int:
        epsilon = self.get_epsilon() if explore else 0.0

        if np.random.random() < epsilon:
            legal = np.where(action_mask > 0)[0]
            return np.random.choice(legal)

        with torch.no_grad():
            obs_dev = obs_to_device(obs, DEVICE)
            q_values = self.behavior_net(obs_dev).squeeze(0)

            # Mask illegal actions
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32).to(DEVICE)
            q_values = q_values + (1 - mask_tensor) * ILLEGAL_ACTION_VALUE

            return q_values.argmax().item()

    def learn(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return 0.0

        batch = self.replay_buffer.sample(BATCH_SIZE)

        states = batch_obs([e.state for e in batch], DEVICE)
        actions = torch.tensor([e.action for e in batch], dtype=torch.long).to(DEVICE)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32).to(DEVICE)
        next_states = batch_obs([e.next_state for e in batch], DEVICE)
        dones = torch.tensor([e.done for e in batch], dtype=torch.float32).to(DEVICE)
        next_masks = torch.tensor(np.stack([e.next_mask for e in batch]),
                                   dtype=torch.float32).to(DEVICE)

        # Current Q values
        q_values = self.behavior_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states)
            next_q = next_q + (1 - next_masks) * ILLEGAL_ACTION_VALUE
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards + DISCOUNT_FACTOR * max_next_q * (1 - dones)

        loss = F.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.behavior_net.parameters(), 1.0)
        self.optimizer.step()

        # Sync target network
        self.total_steps += 1
        if self.total_steps % SYNC_TARGET_STEPS == 0:
            self.target_net.load_state_dict(self.behavior_net.state_dict())

        return loss.item()

    def train_episode(self, env) -> float:
        obs, _ = env.reset(random_start=True)
        total_reward = 0.0

        while not env.done:
            action_mask = env.get_action_mask()
            action = self.select_action(obs, action_mask, explore=True)

            next_obs, reward, done, _, info = env.step(action)
            next_mask = env.get_action_mask()

            self.replay_buffer.push(DQNExperience(
                state=obs, action=action, reward=reward,
                next_state=next_obs, done=done,
                mask=action_mask, next_mask=next_mask
            ))

            self.obs_index += 1

            if len(self.replay_buffer) >= BATCH_SIZE:
                self.learn()

            obs = next_obs
            total_reward += reward

        return total_reward


# ============================================================
# Dueling DQN Agent
# ============================================================

class DuelingDQNAgent:
    def __init__(self, use_images: bool = True, raw_dim: int = 336):
        self.use_images = use_images
        self.behavior_net = DuelingDQNNetwork(use_images, raw_dim).to(DEVICE)
        self.target_net = DuelingDQNNetwork(use_images, raw_dim).to(DEVICE)
        self.target_net.load_state_dict(self.behavior_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.behavior_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer()
        self.obs_index = 0
        self.total_steps = 0

    def get_epsilon(self) -> float:
        return max(EPSILON_END, EPSILON_START - self.obs_index / EPSILON_DECAY_DIVISOR)

    def select_action(self, obs: dict, action_mask: np.ndarray,
                      explore: bool = True) -> int:
        epsilon = self.get_epsilon() if explore else 0.0

        if np.random.random() < epsilon:
            legal = np.where(action_mask > 0)[0]
            return np.random.choice(legal)

        with torch.no_grad():
            obs_dev = obs_to_device(obs, DEVICE)
            q_values = self.behavior_net(obs_dev).squeeze(0)
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32).to(DEVICE)
            q_values = q_values + (1 - mask_tensor) * ILLEGAL_ACTION_VALUE
            return q_values.argmax().item()

    def learn(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return 0.0

        batch = self.replay_buffer.sample(BATCH_SIZE)

        states = batch_obs([e.state for e in batch], DEVICE)
        actions = torch.tensor([e.action for e in batch], dtype=torch.long).to(DEVICE)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32).to(DEVICE)
        next_states = batch_obs([e.next_state for e in batch], DEVICE)
        dones = torch.tensor([e.done for e in batch], dtype=torch.float32).to(DEVICE)
        next_masks = torch.tensor(np.stack([e.next_mask for e in batch]),
                                   dtype=torch.float32).to(DEVICE)

        q_values = self.behavior_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states)
            next_q = next_q + (1 - next_masks) * ILLEGAL_ACTION_VALUE
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards + DISCOUNT_FACTOR * max_next_q * (1 - dones)

        loss = F.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.behavior_net.parameters(), 1.0)
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % SYNC_TARGET_STEPS == 0:
            self.target_net.load_state_dict(self.behavior_net.state_dict())

        return loss.item()

    def train_episode(self, env) -> float:
        obs, _ = env.reset(random_start=True)
        total_reward = 0.0

        while not env.done:
            action_mask = env.get_action_mask()
            action = self.select_action(obs, action_mask, explore=True)

            next_obs, reward, done, _, info = env.step(action)
            next_mask = env.get_action_mask()

            self.replay_buffer.push(DQNExperience(
                state=obs, action=action, reward=reward,
                next_state=next_obs, done=done,
                mask=action_mask, next_mask=next_mask
            ))

            self.obs_index += 1

            if len(self.replay_buffer) >= BATCH_SIZE:
                self.learn()

            obs = next_obs
            total_reward += reward

        return total_reward


# ============================================================
# PPO Agent
# ============================================================

class PPOAgent:
    def __init__(self, use_images: bool = True, raw_dim: int = 336):
        self.use_images = use_images
        self.actor = PPOActorNetwork(use_images, raw_dim).to(DEVICE)
        self.critic = PPOCriticNetwork(use_images, raw_dim).to(DEVICE)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=PPO_ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=PPO_CRITIC_LR)

        self.transition_buffer = []

    def select_action(self, obs: dict, action_mask: np.ndarray) -> Tuple[int, float, float]:
        """Select action and return (action, log_prob, value)."""
        with torch.no_grad():
            obs_dev = obs_to_device(obs, DEVICE)
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32).to(DEVICE)

            logits = self.actor(obs_dev, mask_tensor).squeeze(0)
            probs = F.softmax(logits, dim=-1)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            value = self.critic(obs_dev).squeeze()

        return action.item(), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + DISCOUNT_FACTOR * next_value * (1 - dones[t]) - values[t]
            gae = delta + DISCOUNT_FACTOR * PPO_GAE_LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return torch.tensor(advantages, dtype=torch.float32).to(DEVICE), \
               torch.tensor(returns, dtype=torch.float32).to(DEVICE)

    def learn(self):
        """PPO update from transition buffer."""
        if len(self.transition_buffer) == 0:
            return 0.0

        states = [t.state for t in self.transition_buffer]
        actions = torch.tensor([t.action for t in self.transition_buffer],
                               dtype=torch.long).to(DEVICE)
        old_log_probs = torch.tensor([t.log_prob for t in self.transition_buffer],
                                      dtype=torch.float32).to(DEVICE)
        values = [t.value for t in self.transition_buffer]
        rewards = [t.reward for t in self.transition_buffer]
        dones = [t.done for t in self.transition_buffer]
        masks = torch.tensor(np.stack([t.mask for t in self.transition_buffer]),
                              dtype=torch.float32).to(DEVICE)

        # Compute GAE
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        T = len(self.transition_buffer)

        for _ in range(PPO_EPOCHS):
            # Sample minibatches sequentially
            indices = np.arange(T)
            np.random.shuffle(indices)

            for start in range(0, T, BATCH_SIZE):
                end = min(start + BATCH_SIZE, T)
                if end - start < 2:
                    continue
                mb_idx = indices[start:end]

                mb_states = batch_obs([states[i] for i in mb_idx], DEVICE)
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_masks = masks[mb_idx]

                # Actor loss
                logits = self.actor(mb_states, mb_masks)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPSILON,
                                     1 + PPO_CLIP_EPSILON) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                values_pred = self.critic(mb_states).squeeze(-1)
                critic_loss = F.mse_loss(values_pred, mb_returns)

                # Total loss
                loss = actor_loss + PPO_CRITIC_DISCOUNT * critic_loss - PPO_ENTROPY_BETA * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_loss += loss.item()

        # Clear buffer
        self.transition_buffer.clear()
        return total_loss

    def train_episode(self, env) -> float:
        obs, _ = env.reset(random_start=True)
        total_reward = 0.0
        self.transition_buffer.clear()

        while not env.done:
            action_mask = env.get_action_mask()
            action, log_prob, value = self.select_action(obs, action_mask)

            next_obs, reward, done, _, info = env.step(action)

            self.transition_buffer.append(PPOTransition(
                state=obs, action=action, log_prob=log_prob,
                value=value, reward=reward, done=done,
                mask=action_mask
            ))

            obs = next_obs
            total_reward += reward

        # Learn from collected trajectory
        self.learn()

        return total_reward


def create_agent(model_name: str, use_images: bool = True, raw_dim: int = 336):
    """Factory function to create agents."""
    if model_name == 'DQN':
        return DQNAgent(use_images, raw_dim)
    elif model_name == 'DuelingDQN':
        return DuelingDQNAgent(use_images, raw_dim)
    elif model_name == 'PPO':
        return PPOAgent(use_images, raw_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")
