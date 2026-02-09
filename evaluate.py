"""
Evaluation metrics and visualization.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    ACTION_IDLE, ACTION_LONG, ACTION_SHORT, ACTION_CLOSE,
    INITIAL_BALANCE, RANDOM_POLICY_RUNS
)
from ensemble import compute_sortino_ratio, compute_volatility, compute_max_drawdown

ACTION_NAMES = {0: 'Idle', 1: 'Long', 2: 'Short', 3: 'Close'}


def compute_all_metrics(balance_history, actions, initial_balance=INITIAL_BALANCE):
    """Compute all evaluation metrics."""
    bh = np.array(balance_history)
    cumulative_return = (bh[-1] - initial_balance) / initial_balance * 100 if len(bh) > 0 else 0

    # Step returns for Sortino
    if len(bh) > 1:
        step_returns = np.diff(bh) / np.array(bh[:-1])
    else:
        step_returns = np.array([0.0])

    total = len(actions)
    trading = sum(1 for a in actions if a in [ACTION_LONG, ACTION_SHORT, ACTION_CLOSE])

    return {
        'cumulative_return_pct': cumulative_return,
        'sortino_ratio': compute_sortino_ratio(step_returns),
        'volatility': compute_volatility(balance_history),
        'max_drawdown': compute_max_drawdown(balance_history),
        'trading_coverage_pct': trading / total * 100 if total > 0 else 0,
        'num_long': sum(1 for a in actions if a == ACTION_LONG),
        'num_short': sum(1 for a in actions if a == ACTION_SHORT),
        'num_close': sum(1 for a in actions if a == ACTION_CLOSE),
        'num_idle': sum(1 for a in actions if a == ACTION_IDLE),
        'total_steps': total,
    }


def print_metrics(metrics: Dict, label: str = ""):
    """Print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Cumulative Return:  {metrics['cumulative_return_pct']:.4f}%")
    print(f"  Sortino Ratio:      {metrics['sortino_ratio']:.4f}")
    print(f"  Volatility:         {metrics['volatility']:.4f}")
    print(f"  Max Drawdown:       {metrics['max_drawdown']:.4f}")
    print(f"  Trading Coverage:   {metrics['trading_coverage_pct']:.4f}%")
    print(f"  Actions: Long={metrics['num_long']}, Short={metrics['num_short']}, "
          f"Close={metrics['num_close']}, Idle={metrics['num_idle']}")
    print(f"{'='*60}\n")


def run_buy_and_hold(env):
    """Run buy-and-hold baseline."""
    obs, _ = env.reset(random_start=False)
    # Open long at start
    action_mask = env.get_action_mask()
    obs, r, done, _, _ = env.step(ACTION_LONG)
    actions = [ACTION_LONG]

    while not env.done:
        obs, r, done, _, _ = env.step(ACTION_IDLE)
        actions.append(ACTION_IDLE)

    return env.balance_history, actions


def run_random_policy(env, num_runs=RANDOM_POLICY_RUNS):
    """Run random policy baseline (averaged over multiple runs)."""
    all_returns = []
    for _ in range(num_runs):
        obs, _ = env.reset(random_start=False)
        actions = []
        while not env.done:
            mask = env.get_action_mask()
            legal = np.where(mask > 0)[0]
            action = np.random.choice(legal)
            obs, r, done, _, _ = env.step(action)
            actions.append(action)
        all_returns.append(env.get_cumulative_return())

    return np.mean(all_returns)


def run_heuristic_trb(env, lookback=1, threshold=0.5):
    """Run Trading Range Break heuristic policy."""
    obs, _ = env.reset(random_start=False)
    actions = []
    prices = []

    while not env.done:
        mask = env.get_action_mask()
        current_price = env.base_data.iloc[env.current_step]['close']
        prices.append(current_price)

        if len(prices) > lookback:
            recent = prices[-lookback-1:-1]
            high = max(recent)
            low = min(recent)
            range_mid = (high + low) / 2
            range_half = (high - low) * threshold

            if current_price > range_mid + range_half and ACTION_LONG in env.get_legal_actions():
                action = ACTION_LONG
            elif current_price < range_mid - range_half and ACTION_SHORT in env.get_legal_actions():
                action = ACTION_SHORT
            elif env.position != 0 and ACTION_CLOSE in env.get_legal_actions():
                if (env.position == 1 and current_price < range_mid) or \
                   (env.position == -1 and current_price > range_mid):
                    action = ACTION_CLOSE
                else:
                    action = ACTION_IDLE
            else:
                action = ACTION_IDLE
        else:
            action = ACTION_IDLE

        obs, r, done, _, _ = env.step(action)
        actions.append(action)

    return env.balance_history, actions


def plot_cumulative_returns(results_dict: Dict[str, list], save_path: str, title: str = ""):
    """Plot cumulative returns comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    for label, bh in results_dict.items():
        returns = (np.array(bh) - bh[0]) / bh[0] * 100
        ax.plot(returns, label=label, alpha=0.8)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title(title or 'Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_trading_decisions(prices, actions, save_path, title=""):
    """Plot trading decisions on price chart."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.plot(prices, color='gray', alpha=0.6, linewidth=0.5)

    for i, action in enumerate(actions):
        if action == ACTION_LONG:
            ax.scatter(i, prices[i], marker='^', color='green', s=20, zorder=5)
        elif action == ACTION_SHORT:
            ax.scatter(i, prices[i], marker='v', color='red', s=20, zorder=5)
        elif action == ACTION_CLOSE:
            ax.scatter(i, prices[i], marker='s', color='blue', s=15, zorder=5)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Close Price')
    ax.set_title(title or 'Trading Decisions')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='Long'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10, label='Short'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, label='Close'),
    ]
    ax.legend(handles=legend_elements)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_gain_loss(balance_history, save_path, title="", initial=INITIAL_BALANCE):
    """Plot cumulative gain/loss."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    gains = np.array(balance_history) - initial
    ax.fill_between(range(len(gains)), gains, where=gains >= 0, color='green', alpha=0.3)
    ax.fill_between(range(len(gains)), gains, where=gains < 0, color='red', alpha=0.3)
    ax.plot(gains, color='black', linewidth=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Gain / Loss (USDT)')
    ax.set_title(title or 'Cumulative Gain/Loss')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
