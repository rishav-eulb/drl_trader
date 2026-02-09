"""
Configuration and hyperparameters for the ensemble DRL cryptocurrency trader.
Based on Jing & Kang (2024) - Expert Systems With Applications.
"""

import os

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ============================================================
# Data Configuration
# ============================================================
TRADING_INTERVAL_MIN = 15  # Base trading interval in minutes
RESOLUTIONS = [15, 30, 120]  # Multi-resolution: 15-min, 30-min, 2-hour
MA_PERIODS = [5, 20, 50]  # Moving average periods

# Number of candles displayed in each area of the image
NUM_CANDLES = {
    15: 24,   # 24 candles in 15-min area
    30: 12,   # 12 candles in 30-min area
    120: 12,  # 12 candles in 2-hour area
}

# ============================================================
# Candlestick Image Configuration
# ============================================================
IMAGE_SIZE = 224  # 224x224 RGB image
CANDLE_WIDTH = 6  # Width of each candle in pixels
CANDLE_SPACING = 3  # Space between candles in pixels

# Image layout heights
ROW1_HEIGHT = 90   # Height for 30-min and 2-h candlesticks (top row)
ROW2_HEIGHT = 134  # Height for 15-min candlesticks (bottom row)

# Color encoding (RGB)
BEARISH_COLOR = (255, 0, 0)     # Red channel
BULLISH_COLOR = (0, 255, 0)     # Green channel
DOJI_COLOR = (255, 255, 0)      # Yellow (open == close)

# MA grayscale values (for blue channel)
MA_PIXEL_VALUES = {
    5: 127,
    20: 170,
    50: 212,
}

# Price ratio bar scaling
PRICE_RATIO_DIVISOR = 100000

# ============================================================
# Walk-Forward Validation
# ============================================================
NUM_FOLDERS = 9  # Number of rolling data folders
TRAIN_DAYS = 180  # Training period in days
VAL_DAYS = 80  # Validation period in days
ROLLING_WINDOW_DAYS = 80  # Rolling window step size
FOLDER_DAYS = TRAIN_DAYS + VAL_DAYS  # Total days per folder (260)

# ============================================================
# Trading Environment
# ============================================================
INITIAL_BALANCE = 100000  # Initial account balance in USDT
TRANSACTION_COST = 0.0005  # 0.05% transaction cost
REWARD_SCALE = 100  # Divide reward by this value

# Position encoding
POSITION_SHORT = -1
POSITION_NONE = 0
POSITION_LONG = 1

# Action encoding
ACTION_IDLE = 0
ACTION_LONG = 1
ACTION_SHORT = 2
ACTION_CLOSE = 3
NUM_ACTIONS = 4

# ============================================================
# DRL Common Hyperparameters
# ============================================================
DISCOUNT_FACTOR = 0.99  # gamma
BATCH_SIZE = 64
NUM_EPISODES = 50
LEARNING_RATE = 1e-4
LATEST_DATA_THRESHOLD = 0.2  # Ensure latest 20% of training data is sampled

# ============================================================
# DQN / Dueling DQN Hyperparameters
# ============================================================
REPLAY_BUFFER_SIZE = 10000
SYNC_TARGET_STEPS = 1000  # Steps to update target network
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_DIVISOR = 1e5  # epsilon = obs_index / this value
ILLEGAL_ACTION_VALUE = -1e8  # Large negative for illegal actions

# ============================================================
# PPO Hyperparameters
# ============================================================
PPO_ACTOR_LR = 1e-4
PPO_CRITIC_LR = 1e-3
PPO_CLIP_EPSILON = 0.2  # Clipping parameter
PPO_GAE_LAMBDA = 0.95  # Generalized advantage estimation
PPO_EPOCHS = 10  # Optimization epochs per trajectory
PPO_CRITIC_DISCOUNT = 0.5
PPO_ENTROPY_BETA = 1e-3

# ============================================================
# Network Architecture
# ============================================================
# CNN backbone: ResNet18 (without last 2 layers) + CBAM
RESNET_FEATURE_DIM = 512  # Output dimension of ResNet18 backbone
STATE_DIM = RESNET_FEATURE_DIM + 2  # + funding_rate + position = 514

# MLP layers for image-based models
MLP_HIDDEN1 = 512
MLP_HIDDEN2 = 256

# MLP layers for raw data models (comparison)
MLP_RAW_HIDDEN1 = 10
MLP_RAW_HIDDEN2 = 5

# CBAM parameters
CBAM_REDUCTION = 16
CBAM_KERNEL_SIZE = 7

# ============================================================
# Ensemble Configuration
# ============================================================
NUM_ENSEMBLE_AGENTS = 3  # Top 3 agents by Sortino ratio
FUNDING_RATE_SCALE = 100  # Multiply funding rate by this

# ============================================================
# Evaluation
# ============================================================
RISK_FREE_RATE = 0  # For Sortino ratio calculation
RANDOM_POLICY_RUNS = 10  # Number of runs for random policy baseline

# ============================================================
# Device
# ============================================================
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
