"""
Multi-resolution candlestick image generation.
Creates 224x224 RGB images encoding:
- Red channel: bearish candles
- Green channel: bullish candles
- Blue channel: moving average curves
- Price ratio bar between 30-min and 2-h areas

Layout:
  Top row (90px height): [2-h candles | ratio bar | 30-min candles]
  Bottom row (134px height): [15-min candles]
"""

import numpy as np
from typing import Dict, Optional
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    IMAGE_SIZE, CANDLE_WIDTH, CANDLE_SPACING,
    ROW1_HEIGHT, ROW2_HEIGHT,
    BEARISH_COLOR, BULLISH_COLOR, DOJI_COLOR,
    MA_PIXEL_VALUES, MA_PERIODS,
    PRICE_RATIO_DIVISOR, NUM_CANDLES, RESOLUTIONS
)


def _scale_price_to_pixels(prices: np.ndarray, plot_height: int,
                            price_min: float, price_max: float) -> np.ndarray:
    """Scale price values to pixel coordinates within a plot area."""
    if price_max == price_min:
        return np.full_like(prices, plot_height // 2, dtype=int)

    # Invert Y axis (higher price = lower pixel value)
    scaled = plot_height - 1 - ((prices - price_min) / (price_max - price_min) * (plot_height - 1))
    return np.clip(scaled.astype(int), 0, plot_height - 1)


def _draw_candle(image: np.ndarray, x_start: int, y_offset: int,
                 open_px: int, high_px: int, low_px: int, close_px: int,
                 is_bullish: bool, is_doji: bool):
    """Draw a single candlestick on the image."""
    if is_doji:
        color = DOJI_COLOR
    elif is_bullish:
        color = BULLISH_COLOR
    else:
        color = BEARISH_COLOR

    # Draw wick (high to low) - thin line in center
    wick_x = x_start + CANDLE_WIDTH // 2
    wick_top = min(high_px, low_px) + y_offset
    wick_bottom = max(high_px, low_px) + y_offset

    for y in range(wick_top, wick_bottom + 1):
        if 0 <= y < image.shape[0] and 0 <= wick_x < image.shape[1]:
            image[y, wick_x] = color

    # Draw body (open to close) - full width
    body_top = min(open_px, close_px) + y_offset
    body_bottom = max(open_px, close_px) + y_offset

    if body_top == body_bottom:
        body_bottom = body_top + 1  # Minimum 1px body

    for y in range(body_top, body_bottom + 1):
        for x in range(x_start, x_start + CANDLE_WIDTH):
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                image[y, x] = color


def _draw_candles_in_area(image: np.ndarray, candle_data: pd.DataFrame,
                           x_offset: int, y_offset: int,
                           plot_height: int, plot_width: int):
    """Draw candlesticks in a designated area of the image."""
    if len(candle_data) == 0:
        return

    # Get price range for scaling
    all_prices = np.concatenate([
        candle_data['high'].values,
        candle_data['low'].values
    ])
    price_min = all_prices.min()
    price_max = all_prices.max()

    for i, (_, row) in enumerate(candle_data.iterrows()):
        x_start = x_offset + i * (CANDLE_WIDTH + CANDLE_SPACING)
        if x_start + CANDLE_WIDTH > x_offset + plot_width:
            break

        open_px = _scale_price_to_pixels(
            np.array([row['open']]), plot_height, price_min, price_max)[0]
        high_px = _scale_price_to_pixels(
            np.array([row['high']]), plot_height, price_min, price_max)[0]
        low_px = _scale_price_to_pixels(
            np.array([row['low']]), plot_height, price_min, price_max)[0]
        close_px = _scale_price_to_pixels(
            np.array([row['close']]), plot_height, price_min, price_max)[0]

        is_bullish = row['close'] > row['open']
        is_doji = row['close'] == row['open']

        _draw_candle(image, x_start, y_offset,
                     open_px, high_px, low_px, close_px,
                     is_bullish, is_doji)


def _draw_ma_curves(ma_image: np.ndarray, candle_data: pd.DataFrame,
                     x_offset: int, y_offset: int,
                     plot_height: int, plot_width: int):
    """Draw moving average curves on a grayscale image."""
    if len(candle_data) == 0:
        return

    # Get price range including MAs for proper scaling
    all_vals = [candle_data['high'].values, candle_data['low'].values]
    for period in MA_PERIODS:
        ma_col = f'ma_{period}'
        if ma_col in candle_data.columns:
            vals = candle_data[ma_col].dropna().values
            if len(vals) > 0:
                all_vals.append(vals)

    all_prices = np.concatenate(all_vals)
    price_min = all_prices.min()
    price_max = all_prices.max()

    for period in MA_PERIODS:
        ma_col = f'ma_{period}'
        if ma_col not in candle_data.columns:
            continue

        pixel_value = MA_PIXEL_VALUES[period]
        prev_y = None

        for i, (_, row) in enumerate(candle_data.iterrows()):
            if pd.isna(row[ma_col]):
                prev_y = None
                continue

            x_center = x_offset + i * (CANDLE_WIDTH + CANDLE_SPACING) + CANDLE_WIDTH // 2
            if x_center >= x_offset + plot_width:
                break

            y_px = _scale_price_to_pixels(
                np.array([row[ma_col]]), plot_height, price_min, price_max)[0]
            y_px += y_offset

            # Draw line from previous point to current
            if prev_y is not None:
                prev_x = x_offset + (i - 1) * (CANDLE_WIDTH + CANDLE_SPACING) + CANDLE_WIDTH // 2
                _draw_line(ma_image, prev_x, prev_y, x_center, y_px, pixel_value)
            else:
                if 0 <= y_px < ma_image.shape[0] and 0 <= x_center < ma_image.shape[1]:
                    ma_image[y_px, x_center] = pixel_value

            prev_y = y_px


def _draw_line(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, value: int):
    """Draw a line using Bresenham's algorithm on a 2D image."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= y0 < image.shape[0] and 0 <= x0 < image.shape[1]:
            image[y0, x0] = value

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _draw_ratio_bar(image: np.ndarray, current_close: float,
                     x_start: int, y_offset: int, bar_width: int,
                     plot_height: int):
    """Draw price ratio bar between 30-min and 2-h areas."""
    ratio = current_close / PRICE_RATIO_DIVISOR
    bar_height = int(min(ratio * plot_height, plot_height))
    bar_height = max(bar_height, 1)

    # Draw from bottom of the area upward
    for y in range(plot_height - bar_height, plot_height):
        for x in range(x_start, x_start + bar_width):
            py = y + y_offset
            if 0 <= py < image.shape[0] and 0 <= x < image.shape[1]:
                image[py, x] = (0, 0, 255)  # Blue


def generate_candlestick_image(multi_res_windows: Dict[int, pd.DataFrame],
                                current_close: float) -> np.ndarray:
    """
    Generate a 224x224 RGB multi-resolution candlestick image.

    Args:
        multi_res_windows: Dict mapping resolution (minutes) to DataFrame
                          with OHLC + MA data for the window
        current_close: Current close price for ratio bar

    Returns:
        224x224x3 numpy array (uint8) - the candlestick image

    Layout:
        Top row (0-89): [2-h area (left) | ratio bar | 30-min area (right)]
        Bottom row (90-223): [15-min area (full width)]
    """
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    ma_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    # Calculate widths
    candle_area_width_12 = 12 * (CANDLE_WIDTH + CANDLE_SPACING) - CANDLE_SPACING  # 105 px for 12 candles
    ratio_bar_width = IMAGE_SIZE - 2 * candle_area_width_12  # Remaining space

    # === Top Row: 2-h (left) and 30-min (right) ===
    # 2-hour area (left side)
    if 120 in multi_res_windows and len(multi_res_windows[120]) > 0:
        _draw_candles_in_area(
            image, multi_res_windows[120],
            x_offset=0, y_offset=0,
            plot_height=ROW1_HEIGHT,
            plot_width=candle_area_width_12
        )
        _draw_ma_curves(
            ma_image, multi_res_windows[120],
            x_offset=0, y_offset=0,
            plot_height=ROW1_HEIGHT,
            plot_width=candle_area_width_12
        )

    # Price ratio bar (center)
    _draw_ratio_bar(
        image, current_close,
        x_start=candle_area_width_12,
        y_offset=0,
        bar_width=ratio_bar_width,
        plot_height=ROW1_HEIGHT
    )

    # 30-min area (right side)
    x_30min = candle_area_width_12 + ratio_bar_width
    if 30 in multi_res_windows and len(multi_res_windows[30]) > 0:
        _draw_candles_in_area(
            image, multi_res_windows[30],
            x_offset=x_30min, y_offset=0,
            plot_height=ROW1_HEIGHT,
            plot_width=candle_area_width_12
        )
        _draw_ma_curves(
            ma_image, multi_res_windows[30],
            x_offset=x_30min, y_offset=0,
            plot_height=ROW1_HEIGHT,
            plot_width=candle_area_width_12
        )

    # === Bottom Row: 15-min (full width) ===
    if 15 in multi_res_windows and len(multi_res_windows[15]) > 0:
        _draw_candles_in_area(
            image, multi_res_windows[15],
            x_offset=0, y_offset=ROW1_HEIGHT,
            plot_height=ROW2_HEIGHT,
            plot_width=IMAGE_SIZE
        )
        _draw_ma_curves(
            ma_image, multi_res_windows[15],
            x_offset=0, y_offset=ROW1_HEIGHT,
            plot_height=ROW2_HEIGHT,
            plot_width=IMAGE_SIZE
        )

    # === Merge: Replace blue channel with MA grayscale ===
    # Where there are MA curves, put them in the blue channel
    ma_mask = ma_image > 0
    image[:, :, 2] = np.where(ma_mask, ma_image, image[:, :, 2])

    return image


def generate_image_for_timestamp(multi_res: Dict[int, pd.DataFrame],
                                  timestamp: pd.Timestamp) -> Optional[np.ndarray]:
    """
    Generate candlestick image for a specific timestamp.

    Args:
        multi_res: Full multi-resolution data dict
        timestamp: The current timestamp

    Returns:
        224x224x3 image or None if insufficient data
    """
    windows = {}
    for res in RESOLUTIONS:
        df = multi_res[res]
        mask = df.index <= timestamp
        num_candles = NUM_CANDLES[res]
        window = df[mask].tail(num_candles)

        if len(window) < num_candles:
            return None  # Not enough data

        windows[res] = window

    # Current close price from 15-min resolution
    current_close = windows[RESOLUTIONS[0]].iloc[-1]['close']

    return generate_candlestick_image(windows, current_close)


if __name__ == "__main__":
    print("Candlestick image generation module loaded successfully.")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Layout: Row1={ROW1_HEIGHT}px (2h+30m), Row2={ROW2_HEIGHT}px (15m)")
