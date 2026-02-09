"""
Trading Environment - OpenAI Gym compatible.
Implements the MDP formulation from the paper:
- State: [candlestick_features (512x1), funding_rate, position]
- Actions: idle(0), long(1), short(2), close(3)
- Reward: balance difference with transaction costs
- Action masking for legal actions
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    NUM_ACTIONS, ACTION_IDLE, ACTION_LONG, ACTION_SHORT, ACTION_CLOSE,
    POSITION_SHORT, POSITION_NONE, POSITION_LONG,
    INITIAL_BALANCE, TRANSACTION_COST, REWARD_SCALE,
    RESNET_FEATURE_DIM, STATE_DIM, LATEST_DATA_THRESHOLD,
    RESOLUTIONS, NUM_CANDLES, FUNDING_RATE_SCALE
)
from utils.candlestick_image import generate_image_for_timestamp


class CryptoTradingEnv(gym.Env):
    """
    Cryptocurrency trading environment for DRL agents.

    The environment provides multi-resolution candlestick images as observations
    and accepts discrete trading actions.
    """

    def __init__(self,
                 multi_res_data: Dict[int, pd.DataFrame],
                 start_date: pd.Timestamp,
                 end_date: pd.Timestamp,
                 use_images: bool = True,
                 initial_balance: float = INITIAL_BALANCE,
                 transaction_cost: float = TRANSACTION_COST):
        super().__init__()

        self.multi_res = multi_res_data
        self.use_images = use_images
        self.initial_balance = initial_balance
        self.tc = transaction_cost

        # Get the base resolution (15-min) timestamps within the date range
        base_df = multi_res_data[RESOLUTIONS[0]]
        mask = (base_df.index >= start_date) & (base_df.index <= end_date)
        self.timestamps = base_df.index[mask].tolist()
        self.base_data = base_df.loc[mask]

        if len(self.timestamps) == 0:
            raise ValueError(f"No data found between {start_date} and {end_date}")

        # Minimum start index to ensure enough candles for image generation
        self.min_start_idx = 0  # Already handled in data preparation

        # Latest data threshold
        self.latest_threshold_idx = int(len(self.timestamps) * (1 - LATEST_DATA_THRESHOLD))

        # Action and observation spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        if use_images:
            # Image observation: 224x224x3
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
                'funding_rate': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                'position': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            })
        else:
            # Raw data observation
            raw_dim = sum(7 * NUM_CANDLES[r] for r in RESOLUTIONS)  # 7 features per candle
            self.observation_space = spaces.Dict({
                'raw_data': spaces.Box(low=-np.inf, high=np.inf, shape=(raw_dim,), dtype=np.float32),
                'funding_rate': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                'position': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            })

        # State variables
        self.current_step = 0
        self.position = POSITION_NONE
        self.balance = initial_balance
        self.coins_held = 0.0
        self.done = False

        # Tracking
        self.trade_history = []
        self.balance_history = []

    def reset(self, random_start: bool = True, seed=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        self.position = POSITION_NONE
        self.balance = self.initial_balance
        self.coins_held = 0.0
        self.done = False
        self.trade_history = []
        self.balance_history = [self.balance]

        if random_start:
            # Ensure latest data threshold is respected
            max_start = self.latest_threshold_idx
            self.current_step = np.random.randint(0, max(1, max_start))
        else:
            self.current_step = 0

        obs = self._get_observation()
        return obs, {}

    def step(self, action: int) -> Tuple:
        """Execute one trading step."""
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        # Mask illegal actions
        legal_actions = self.get_legal_actions()
        if action not in legal_actions:
            action = ACTION_IDLE  # Default to idle if illegal

        # Get current and next close prices
        ct = self.base_data.iloc[self.current_step]['close']

        # Check if next step exists
        if self.current_step + 1 >= len(self.timestamps):
            self.done = True
            # Force close position at end
            if self.position != POSITION_NONE:
                action = ACTION_CLOSE

        ct1 = self.base_data.iloc[min(self.current_step + 1, len(self.timestamps) - 1)]['close']

        # Calculate reward
        reward = self._calculate_reward(action, ct, ct1)

        # Execute action
        self._execute_action(action, ct)

        # Record
        self.trade_history.append({
            'step': self.current_step,
            'timestamp': self.timestamps[self.current_step],
            'action': action,
            'position': self.position,
            'balance': self.balance,
            'price': ct,
        })
        self.balance_history.append(self._get_portfolio_value(ct))

        # Advance step
        self.current_step += 1
        if self.current_step >= len(self.timestamps):
            self.done = True

        obs = self._get_observation() if not self.done else self._get_observation_safe()
        info = {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self._get_portfolio_value(ct1),
        }

        return obs, reward / REWARD_SCALE, self.done, False, info

    def _calculate_reward(self, action: int, ct: float, ct1: float) -> float:
        """Calculate reward based on action and price change."""
        ht = self.coins_held
        tc = self.tc

        if action == ACTION_IDLE:
            if self.position == POSITION_NONE:
                return 0.0
            elif self.position == POSITION_LONG:
                return ht * (ct1 - ct)
            elif self.position == POSITION_SHORT:
                return -ht * (ct1 - ct)

        elif action == ACTION_LONG:
            if self.position == POSITION_NONE:
                new_balance = self.balance * (1 - tc)
                new_coins = new_balance / ct
                return new_coins * (ct1 - ct)

        elif action == ACTION_SHORT:
            if self.position == POSITION_NONE:
                new_coins = self.balance * (1 - tc) / ct
                return -new_coins * (ct1 - ct)

        elif action == ACTION_CLOSE:
            if self.position == POSITION_LONG:
                return -ht * (ct1 - ct) - tc * ht * ct
            elif self.position == POSITION_SHORT:
                return ht * (ct1 - ct) - tc * ht * ct

        return 0.0

    def _execute_action(self, action: int, ct: float):
        """Execute the trading action and update state."""
        if action == ACTION_LONG and self.position == POSITION_NONE:
            investable = self.balance * (1 - self.tc)
            self.coins_held = investable / ct
            self.balance = 0.0  # All cash spent on coins
            self.position = POSITION_LONG

        elif action == ACTION_SHORT and self.position == POSITION_NONE:
            self.coins_held = self.balance * (1 - self.tc) / ct
            self.balance = self.balance + self.balance * (1 - self.tc)
            self.position = POSITION_SHORT

        elif action == ACTION_CLOSE:
            if self.position == POSITION_LONG:
                # balance is 0 during long; sell coins to get cash back
                self.balance = self.coins_held * ct * (1 - self.tc)
                self.coins_held = 0.0
                self.position = POSITION_NONE
            elif self.position == POSITION_SHORT:
                # balance includes margin + short proceeds; buy back coins
                self.balance = self.balance - self.coins_held * ct * (1 - self.tc)
                self.coins_held = 0.0
                self.position = POSITION_NONE

    def get_legal_actions(self) -> list:
        """Get legal actions based on current position."""
        if self.position == POSITION_NONE:
            return [ACTION_IDLE, ACTION_LONG, ACTION_SHORT]
        else:  # POSITION_LONG or POSITION_SHORT
            return [ACTION_IDLE, ACTION_CLOSE]

    def get_action_mask(self) -> np.ndarray:
        """Get action mask (1 = legal, 0 = illegal)."""
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a in self.get_legal_actions():
            mask[a] = 1.0
        return mask

    def _get_observation(self):
        """Get current observation."""
        if self.current_step >= len(self.timestamps):
            return self._get_observation_safe()

        timestamp = self.timestamps[self.current_step]

        # Funding rate
        fr_col = 'funding_rate'
        if fr_col in self.base_data.columns:
            funding_rate = self.base_data.iloc[self.current_step].get(fr_col, 0.0)
        else:
            funding_rate = 0.0

        if self.use_images:
            image = generate_image_for_timestamp(self.multi_res, timestamp)
            if image is None:
                image = np.zeros((224, 224, 3), dtype=np.uint8)

            return {
                'image': image,
                'funding_rate': np.array([funding_rate], dtype=np.float32),
                'position': np.array([float(self.position)], dtype=np.float32),
            }
        else:
            raw_vector = self._get_raw_vector(timestamp)
            return {
                'raw_data': raw_vector,
                'funding_rate': np.array([funding_rate], dtype=np.float32),
                'position': np.array([float(self.position)], dtype=np.float32),
            }

    def _get_observation_safe(self):
        """Get a safe observation when at the end of data."""
        if self.use_images:
            return {
                'image': np.zeros((224, 224, 3), dtype=np.uint8),
                'funding_rate': np.array([0.0], dtype=np.float32),
                'position': np.array([float(self.position)], dtype=np.float32),
            }
        else:
            raw_dim = sum(7 * NUM_CANDLES[r] for r in RESOLUTIONS)
            return {
                'raw_data': np.zeros(raw_dim, dtype=np.float32),
                'funding_rate': np.array([0.0], dtype=np.float32),
                'position': np.array([float(self.position)], dtype=np.float32),
            }

    def _get_raw_vector(self, timestamp: pd.Timestamp) -> np.ndarray:
        """Get multi-resolution raw data vector for comparison models."""
        vectors = []
        for res in RESOLUTIONS:
            df = self.multi_res[res]
            mask = df.index <= timestamp
            num_candles = NUM_CANDLES[res]
            window = df[mask].tail(num_candles)

            for _, row in window.iterrows():
                vec = [row['open'], row['high'], row['low'], row['close']]
                for p in [5, 20, 50]:
                    ma_col = f'ma_{p}'
                    vec.append(row.get(ma_col, 0.0) if not pd.isna(row.get(ma_col, np.nan)) else 0.0)
                vectors.extend(vec)

        # Pad if necessary
        expected_dim = sum(7 * NUM_CANDLES[r] for r in RESOLUTIONS)
        while len(vectors) < expected_dim:
            vectors.append(0.0)

        return np.array(vectors[:expected_dim], dtype=np.float32)

    def _get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value."""
        if self.position == POSITION_LONG:
            return self.balance + self.coins_held * current_price
        elif self.position == POSITION_SHORT:
            return self.balance - self.coins_held * current_price
        return self.balance

    def get_cumulative_return(self) -> float:
        """Calculate cumulative return."""
        if len(self.balance_history) < 2:
            return 0.0
        return (self.balance_history[-1] - self.initial_balance) / self.initial_balance

    @property
    def num_steps(self):
        return len(self.timestamps)
