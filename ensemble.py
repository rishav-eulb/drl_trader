"""
Ensemble module: Sortino-ratio weighted voting approach.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    NUM_ENSEMBLE_AGENTS, NUM_ACTIONS, RISK_FREE_RATE,
    ACTION_IDLE, ACTION_LONG, ACTION_SHORT, ACTION_CLOSE, DEVICE
)
from agents import PPOAgent, obs_to_device


def compute_sortino_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    if len(returns) == 0:
        return float('nan')
    excess = returns - risk_free_rate
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float('nan')
    ds_std = np.std(downside, ddof=1)
    if ds_std == 0:
        return float('nan')
    return np.mean(excess) / ds_std


def evaluate_agent(agent, env):
    obs, _ = env.reset(random_start=False)
    step_returns, actions_taken, trading_actions = [], [], 0

    while not env.done:
        action_mask = env.get_action_mask()
        if isinstance(agent, PPOAgent):
            action, _, _ = agent.select_action(obs, action_mask)
        else:
            action = agent.select_action(obs, action_mask, explore=False)

        next_obs, reward, done, _, info = env.step(action)
        step_returns.append(reward)
        actions_taken.append(action)
        if action in [ACTION_LONG, ACTION_SHORT, ACTION_CLOSE]:
            trading_actions += 1
        obs = next_obs

    step_returns = np.array(step_returns)
    total = len(actions_taken)
    return {
        'sortino_ratio': compute_sortino_ratio(step_returns),
        'cumulative_return': env.get_cumulative_return(),
        'trading_coverage': trading_actions / total if total > 0 else 0.0,
        'step_returns': step_returns,
        'balance_history': env.balance_history,
        'actions': actions_taken,
    }


def compute_volatility(bh):
    if len(bh) < 2:
        return 0.0
    r = np.diff(bh) / np.array(bh[:-1])
    return float(np.std(r, ddof=1)) if len(r) > 1 else 0.0


def compute_max_drawdown(bh):
    if len(bh) < 2:
        return 0.0
    b = np.array(bh)
    peak = np.maximum.accumulate(b)
    dd = (b - peak) / peak
    return float(dd.min())


class EnsembleTrader:
    def __init__(self, agents, sortino_ratios, num_ensemble=NUM_ENSEMBLE_AGENTS):
        self.all_agents = agents
        self.all_sortinos = sortino_ratios

        valid = [(i, sr) for i, sr in enumerate(sortino_ratios) if not np.isnan(sr)]
        if len(valid) == 0:
            self.selected_indices = list(range(min(num_ensemble, len(agents))))
            self.weights = np.ones(len(self.selected_indices)) / len(self.selected_indices)
        else:
            valid.sort(key=lambda x: x[1], reverse=True)
            top_k = valid[:num_ensemble]
            self.selected_indices = [i for i, _ in top_k]
            self.weights = np.array([sr for _, sr in top_k])
            if np.any(self.weights < 0):
                self.weights = self.weights - self.weights.min() + 1e-6
            w_sum = self.weights.sum()
            self.weights = self.weights / w_sum if w_sum > 0 else np.ones(len(top_k)) / len(top_k)

        self.selected_agents = [agents[i] for i in self.selected_indices]
        print(f"Ensemble agents: {self.selected_indices}")
        print(f"Sortinos: {[sortino_ratios[i] for i in self.selected_indices]}")
        print(f"Weights: {self.weights.tolist()}")

    def get_ensemble_action(self, obs, action_mask):
        votes = np.zeros(NUM_ACTIONS)
        for agent, w in zip(self.selected_agents, self.weights):
            if isinstance(agent, PPOAgent):
                a, _, _ = agent.select_action(obs, action_mask)
            else:
                a = agent.select_action(obs, action_mask, explore=False)
            votes[a] += w
        votes[action_mask == 0] = -np.inf
        if np.all(np.isinf(votes)):
            return ACTION_IDLE
        return int(np.argmax(votes))

    def get_individual_actions(self, obs, action_mask):
        actions = []
        for agent in self.selected_agents:
            if isinstance(agent, PPOAgent):
                a, _, _ = agent.select_action(obs, action_mask)
            else:
                a = agent.select_action(obs, action_mask, explore=False)
            actions.append(a)
        return actions

    def run_test(self, env):
        obs, _ = env.reset(random_start=False)
        step_returns, actions_taken = [], []
        individual_log, trading_actions = [], 0

        while not env.done:
            mask = env.get_action_mask()
            ind = self.get_individual_actions(obs, mask)
            action = self.get_ensemble_action(obs, mask)
            if mask[action] == 0:
                action = ACTION_IDLE

            next_obs, reward, done, _, info = env.step(action)
            step_returns.append(reward)
            actions_taken.append(action)
            individual_log.append(ind)
            if action in [ACTION_LONG, ACTION_SHORT, ACTION_CLOSE]:
                trading_actions += 1
            obs = next_obs

        sr = np.array(step_returns)
        total = len(actions_taken)
        return {
            'sortino_ratio': compute_sortino_ratio(sr),
            'cumulative_return': env.get_cumulative_return(),
            'trading_coverage': trading_actions / total if total > 0 else 0.0,
            'volatility': compute_volatility(env.balance_history),
            'max_drawdown': compute_max_drawdown(env.balance_history),
            'step_returns': sr,
            'balance_history': env.balance_history,
            'actions': actions_taken,
            'individual_actions': individual_log,
        }
