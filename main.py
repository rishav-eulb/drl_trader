"""
Main entry point: Full pipeline for ensemble DRL cryptocurrency trading.
Usage: python main.py --ohlcv_path data/btc_1min.csv --funding_path data/funding_rate.csv
"""

import argparse
import os
import sys
import copy
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils.data_preprocessing import (
    load_ohlcv, load_funding_rate, prepare_multi_resolution_data,
    create_walk_forward_folders, get_test_period
)
from envs.trading_env import CryptoTradingEnv
from agents import create_agent
from ensemble import EnsembleTrader, evaluate_agent, compute_sortino_ratio
from evaluate import (
    compute_all_metrics, print_metrics, plot_cumulative_returns,
    plot_trading_decisions, plot_gain_loss,
    run_buy_and_hold, run_random_policy, run_heuristic_trb
)


def train_single_agent(model_name, env, num_episodes=NUM_EPISODES, use_images=True):
    """Train a single agent on one data folder."""
    raw_dim = sum(7 * NUM_CANDLES[r] for r in RESOLUTIONS)
    agent = create_agent(model_name, use_images=use_images, raw_dim=raw_dim)

    best_reward = -float('inf')
    best_state = None

    for ep in range(num_episodes):
        reward = agent.train_episode(env)
        if reward > best_reward:
            best_reward = reward
            if hasattr(agent, 'behavior_net'):
                best_state = copy.deepcopy(agent.behavior_net.state_dict())
            elif hasattr(agent, 'actor'):
                best_state = {
                    'actor': copy.deepcopy(agent.actor.state_dict()),
                    'critic': copy.deepcopy(agent.critic.state_dict()),
                }

        if (ep + 1) % 10 == 0:
            print(f"    Episode {ep+1}/{num_episodes} | Reward: {reward:.2f} | Best: {best_reward:.2f}")

    # Load best model
    if best_state is not None:
        if hasattr(agent, 'behavior_net'):
            agent.behavior_net.load_state_dict(best_state)
        elif hasattr(agent, 'actor'):
            agent.actor.load_state_dict(best_state['actor'])
            agent.critic.load_state_dict(best_state['critic'])

    return agent, best_reward


def train_all_folders(model_name, multi_res, folders, use_images=True):
    """Train agents across all walk-forward folders."""
    agents = []
    sortino_ratios = []

    for folder in folders:
        idx = folder['folder_idx']
        print(f"\n--- Folder {idx}/{len(folders)} [{model_name}] ---")
        print(f"    Train: {folder['train_start'].date()} to {folder['train_end'].date()}")
        print(f"    Val:   {folder['val_start'].date()} to {folder['val_end'].date()}")

        # Create training environment
        train_env = CryptoTradingEnv(
            multi_res, folder['train_start'], folder['train_end'],
            use_images=use_images
        )

        # Train
        agent, best_reward = train_single_agent(model_name, train_env,
                                                 use_images=use_images)

        # Validate
        val_env = CryptoTradingEnv(
            multi_res, folder['val_start'], folder['val_end'],
            use_images=use_images
        )
        val_result = evaluate_agent(agent, val_env)
        sr = val_result['sortino_ratio']

        print(f"    Val Sortino: {sr:.4f} | "
              f"Val Return: {val_result['cumulative_return']*100:.2f}% | "
              f"Coverage: {val_result['trading_coverage']*100:.2f}%")

        agents.append(agent)
        sortino_ratios.append(sr)

    return agents, sortino_ratios


def run_pipeline(ohlcv_path, funding_path=None, test_days=30,
                 use_images=True, models_to_train=None):
    """Run the full training and evaluation pipeline."""

    if models_to_train is None:
        models_to_train = ['DQN', 'DuelingDQN', 'PPO']

    # ============================================================
    # 1. Load and preprocess data
    # ============================================================
    print("\n" + "="*60)
    print("  Step 1: Loading and preprocessing data")
    print("="*60)

    df_1min = load_ohlcv(ohlcv_path)
    print(f"Loaded {len(df_1min)} rows of 1-min OHLCV data")
    print(f"Date range: {df_1min.index[0]} to {df_1min.index[-1]}")

    funding = None
    if funding_path and os.path.exists(funding_path):
        funding = load_funding_rate(funding_path)
        print(f"Loaded {len(funding)} funding rate entries")

    multi_res = prepare_multi_resolution_data(df_1min, funding)
    for res, df in multi_res.items():
        print(f"  {res}-min: {len(df)} candles")

    # ============================================================
    # 2. Create walk-forward folders
    # ============================================================
    print("\n" + "="*60)
    print("  Step 2: Creating walk-forward folders")
    print("="*60)

    base_df = multi_res[TRADING_INTERVAL_MIN]
    folders = create_walk_forward_folders(base_df)
    print(f"Created {len(folders)} folders")
    for f in folders:
        print(f"  Folder {f['folder_idx']}: "
              f"Train {f['train_start'].date()}-{f['train_end'].date()} | "
              f"Val {f['val_start'].date()}-{f['val_end'].date()}")

    test_period = get_test_period(folders, base_df, test_days)
    print(f"\nTest period: {test_period['test_start'].date()} to {test_period['test_end'].date()}")

    # ============================================================
    # 3. Train agents for each model
    # ============================================================
    print("\n" + "="*60)
    print("  Step 3: Training DRL agents")
    print("="*60)

    all_results = {}
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

    data_label = "CI" if use_images else "MR"

    for model_name in models_to_train:
        print(f"\n{'#'*60}")
        print(f"  Training {model_name} ({data_label})")
        print(f"{'#'*60}")

        agents, sortinos = train_all_folders(model_name, multi_res, folders,
                                              use_images=use_images)

        print(f"\nSortino ratios: {[f'{s:.4f}' for s in sortinos]}")

        # ============================================================
        # 4. Create ensemble and test
        # ============================================================
        print(f"\n  Creating ensemble for {model_name}...")
        ensemble = EnsembleTrader(agents, sortinos)

        test_env = CryptoTradingEnv(
            multi_res,
            test_period['test_start'], test_period['test_end'],
            use_images=use_images
        )

        test_result = ensemble.run_test(test_env)
        metrics = compute_all_metrics(
            test_result['balance_history'],
            test_result['actions']
        )
        print_metrics(metrics, f"{model_name} ({data_label}) - Ensemble Test")

        all_results[f"{model_name}({data_label})"] = {
            'metrics': metrics,
            'balance_history': test_result['balance_history'],
            'actions': test_result['actions'],
            'ensemble': ensemble,
        }

        # Save model
        save_path = os.path.join(MODEL_DIR, f"{model_name}_{data_label}_ensemble.pt")
        torch.save({
            'sortino_ratios': sortinos,
            'selected_indices': ensemble.selected_indices,
            'weights': ensemble.weights.tolist(),
        }, save_path)
        print(f"  Saved ensemble info: {save_path}")

    # ============================================================
    # 5. Run baselines
    # ============================================================
    print("\n" + "="*60)
    print("  Step 5: Running baselines")
    print("="*60)

    # Buy and Hold
    bh_env = CryptoTradingEnv(multi_res, test_period['test_start'],
                               test_period['test_end'], use_images=use_images)
    bh_history, bh_actions = run_buy_and_hold(bh_env)
    bh_metrics = compute_all_metrics(bh_history, bh_actions)
    print_metrics(bh_metrics, "Buy & Hold Baseline")
    all_results['BuyHold'] = {'metrics': bh_metrics, 'balance_history': bh_history,
                               'actions': bh_actions}

    # Heuristic TRB
    hp_env = CryptoTradingEnv(multi_res, test_period['test_start'],
                               test_period['test_end'], use_images=use_images)
    hp_history, hp_actions = run_heuristic_trb(hp_env)
    hp_metrics = compute_all_metrics(hp_history, hp_actions)
    print_metrics(hp_metrics, "Heuristic TRB Baseline")
    all_results['HeuristicTRB'] = {'metrics': hp_metrics, 'balance_history': hp_history,
                                    'actions': hp_actions}

    # Random Policy (average)
    rp_env = CryptoTradingEnv(multi_res, test_period['test_start'],
                               test_period['test_end'], use_images=use_images)
    rp_avg_return = run_random_policy(rp_env)
    print(f"Random Policy avg cumulative return: {rp_avg_return*100:.4f}%")

    # ============================================================
    # 6. Generate plots
    # ============================================================
    print("\n" + "="*60)
    print("  Step 6: Generating plots")
    print("="*60)

    results_dir = os.path.join(BASE_DIR, 'results')

    # Cumulative returns comparison
    bh_dict = {k: v['balance_history'] for k, v in all_results.items()
                if 'balance_history' in v}
    plot_cumulative_returns(bh_dict,
                           os.path.join(results_dir, 'cumulative_returns.png'),
                           'Cumulative Returns Comparison')

    # Per-model gain/loss and decision plots
    for name, res in all_results.items():
        safe_name = name.replace('(', '_').replace(')', '')
        plot_gain_loss(res['balance_history'],
                      os.path.join(results_dir, f'gain_loss_{safe_name}.png'),
                      f'Gain/Loss: {name}')

        if 'actions' in res and name not in ['BuyHold']:
            test_env_temp = CryptoTradingEnv(
                multi_res, test_period['test_start'], test_period['test_end'],
                use_images=use_images
            )
            prices = test_env_temp.base_data['close'].values[:len(res['actions'])]
            plot_trading_decisions(prices, res['actions'],
                                 os.path.join(results_dir, f'decisions_{safe_name}.png'),
                                 f'Decisions: {name}')

    # ============================================================
    # 7. Summary table
    # ============================================================
    print("\n" + "="*80)
    print("  RESULTS SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Return%':>10} {'Sortino':>10} {'Volatility':>10} "
          f"{'MDD':>10} {'Coverage%':>10}")
    print("-"*80)
    for name, res in all_results.items():
        m = res['metrics']
        print(f"{name:<25} {m['cumulative_return_pct']:>10.4f} "
              f"{m['sortino_ratio']:>10.4f} {m['volatility']:>10.4f} "
              f"{m['max_drawdown']:>10.4f} {m['trading_coverage_pct']:>10.4f}")
    print("="*80)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ensemble DRL Crypto Trader')
    parser.add_argument('--ohlcv_path', type=str, required=True,
                        help='Path to 1-min OHLCV CSV')
    parser.add_argument('--funding_path', type=str, default=None,
                        help='Path to funding rate CSV')
    parser.add_argument('--test_days', type=int, default=30,
                        help='Number of test days (default: 30)')
    parser.add_argument('--use_images', action='store_true', default=True,
                        help='Use candlestick images (default: True)')
    parser.add_argument('--use_raw', action='store_true', default=False,
                        help='Use raw numerical data instead of images')
    parser.add_argument('--models', nargs='+', default=['DQN', 'DuelingDQN', 'PPO'],
                        help='Models to train')

    args = parser.parse_args()

    use_images = not args.use_raw

    print(f"Device: {DEVICE}")
    print(f"Data type: {'Candlestick Images' if use_images else 'Raw Numerical'}")
    print(f"Models: {args.models}")

    results = run_pipeline(
        ohlcv_path=args.ohlcv_path,
        funding_path=args.funding_path,
        test_days=args.test_days,
        use_images=use_images,
        models_to_train=args.models,
    )
