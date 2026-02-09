"""
Data preprocessing: resample 1-min OHLCV data to multi-resolution,
compute moving averages, and prepare walk-forward data folders.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RESOLUTIONS, MA_PERIODS, NUM_CANDLES, NUM_FOLDERS,
    TRAIN_DAYS, VAL_DAYS, ROLLING_WINDOW_DAYS, TRADING_INTERVAL_MIN,
    FUNDING_RATE_SCALE
)


def load_ohlcv(filepath: str) -> pd.DataFrame:
    """Load 1-minute OHLCV data from CSV."""
    df = pd.read_csv(filepath)

    # Handle different column naming conventions
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ['timestamp', 'time', 'date', 'datetime', 'open_time']:
            col_map[col] = 'timestamp'
        elif cl == 'open':
            col_map[col] = 'open'
        elif cl == 'high':
            col_map[col] = 'high'
        elif cl == 'low':
            col_map[col] = 'low'
        elif cl == 'close':
            col_map[col] = 'close'
        elif cl in ['volume', 'vol']:
            col_map[col] = 'volume'

    df = df.rename(columns=col_map)

    # Convert timestamp to datetime
    if df['timestamp'].dtype in ['int64', 'float64']:
        # Millisecond timestamps from Binance
        if df['timestamp'].iloc[0] > 1e12:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.set_index('timestamp').sort_index()

    # Ensure numeric
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    return df


def load_funding_rate(filepath: str) -> pd.Series:
    """Load funding rate data from CSV."""
    df = pd.read_csv(filepath)

    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ['timestamp', 'time', 'date', 'datetime', 'calc_time', 'funding_time']:
            col_map[col] = 'timestamp'
        elif cl in ['funding_rate', 'fundingrate', 'rate', 'funding']:
            col_map[col] = 'funding_rate'

    df = df.rename(columns=col_map)

    if df['timestamp'].dtype in ['int64', 'float64']:
        if df['timestamp'].iloc[0] > 1e12:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.set_index('timestamp').sort_index()
    df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')

    return df['funding_rate']


def resample_ohlcv(df_1min: pd.DataFrame, interval_min: int) -> pd.DataFrame:
    """Resample 1-minute OHLCV data to a given interval."""
    rule = f'{interval_min}min' if interval_min < 60 else f'{interval_min // 60}h'

    resampled = df_1min.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    })

    if 'volume' in df_1min.columns:
        vol = df_1min['volume'].resample(rule).sum()
        resampled['volume'] = vol

    resampled = resampled.dropna(subset=['open', 'high', 'low', 'close'])
    return resampled


def compute_moving_averages(df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
    """Compute moving averages for close prices."""
    if periods is None:
        periods = MA_PERIODS

    for period in periods:
        df[f'ma_{period}'] = df['close'].rolling(window=period).mean()

    return df


def prepare_multi_resolution_data(df_1min: pd.DataFrame,
                                   funding_rate: Optional[pd.Series] = None
                                   ) -> Dict[int, pd.DataFrame]:
    """
    Prepare multi-resolution data from 1-min OHLCV.

    Returns dict: {resolution_minutes: DataFrame with OHLC + MAs}
    """
    multi_res = {}

    for res in RESOLUTIONS:
        if res == 1:
            resampled = df_1min.copy()
        else:
            resampled = resample_ohlcv(df_1min, res)

        resampled = compute_moving_averages(resampled, MA_PERIODS)
        multi_res[res] = resampled

    # Merge funding rate into the base resolution (15-min)
    base_res = TRADING_INTERVAL_MIN
    if funding_rate is not None:
        # Forward-fill funding rate to match 15-min intervals
        funding_reindexed = funding_rate.reindex(
            multi_res[base_res].index, method='ffill'
        )
        multi_res[base_res]['funding_rate'] = funding_reindexed * FUNDING_RATE_SCALE
        multi_res[base_res]['funding_rate'] = multi_res[base_res]['funding_rate'].fillna(0)

    return multi_res


def create_walk_forward_folders(base_df: pd.DataFrame,
                                 num_folders: int = NUM_FOLDERS,
                                 train_days: int = TRAIN_DAYS,
                                 val_days: int = VAL_DAYS,
                                 rolling_days: int = ROLLING_WINDOW_DAYS
                                 ) -> List[Dict[str, Tuple]]:
    """
    Create walk-forward validation folders.

    Each folder has:
    - train_start, train_end dates
    - val_start, val_end dates

    Returns list of dicts with date ranges.
    """
    folders = []
    dates = base_df.index

    # Find the first valid date (after enough MA warmup)
    max_candles_needed = max(NUM_CANDLES.values()) + max(MA_PERIODS)
    first_valid_idx = max_candles_needed
    if first_valid_idx >= len(dates):
        raise ValueError("Not enough data for the required lookback period")

    start_date = dates[first_valid_idx]

    for i in range(num_folders):
        offset = pd.Timedelta(days=rolling_days * i)
        train_start = start_date + offset
        train_end = train_start + pd.Timedelta(days=train_days)
        val_start = train_end
        val_end = val_start + pd.Timedelta(days=val_days)

        # Check if we have enough data
        if val_end > dates[-1]:
            print(f"Warning: Only {i} folders possible with available data (requested {num_folders})")
            break

        folders.append({
            'folder_idx': i + 1,
            'train_start': train_start,
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end,
        })

    return folders


def get_test_period(folders: List[Dict], base_df: pd.DataFrame,
                    test_days: int = 30) -> Dict:
    """Get test period dates following the last validation set."""
    last_folder = folders[-1]
    test_start = last_folder['val_end']
    test_end = test_start + pd.Timedelta(days=test_days)

    # Clip to available data
    if test_end > base_df.index[-1]:
        test_end = base_df.index[-1]

    return {
        'test_start': test_start,
        'test_end': test_end,
    }


def get_candle_window(multi_res: Dict[int, pd.DataFrame],
                      timestamp: pd.Timestamp,
                      resolution: int) -> pd.DataFrame:
    """
    Get the window of candles ending at or before the given timestamp
    for a specific resolution.
    """
    df = multi_res[resolution]
    mask = df.index <= timestamp
    num_candles = NUM_CANDLES[resolution]

    relevant = df[mask].tail(num_candles)
    return relevant


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Data preprocessing module loaded successfully.")
    print(f"Resolutions: {RESOLUTIONS}")
    print(f"MA periods: {MA_PERIODS}")
    print(f"Walk-forward: {NUM_FOLDERS} folders, {TRAIN_DAYS}d train, {VAL_DAYS}d val")
