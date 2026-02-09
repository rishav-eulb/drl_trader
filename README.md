# Ensemble Deep Reinforcement Learning Cryptocurrency Trader

Implementation of "Automated cryptocurrency trading approach using ensemble deep
reinforcement learning: Learn to understand candlesticks" (Jing & Kang, 2024).

## Project Structure

```
crypto_trader/
├── config.py                  # All hyperparameters and configuration
├── main.py                    # Main entry point - runs full pipeline
├── data/
│   └── (place your CSV files here)
├── images/                    # Generated candlestick images (auto-created)
├── models/                    # Saved model checkpoints (auto-created)
├── utils/
│   ├── data_preprocessing.py  # Resample 1-min data, compute MAs
│   └── candlestick_image.py   # Generate 224x224 multi-resolution images
├── envs/
│   └── trading_env.py         # OpenAI Gym trading environment (MDP)
├── networks.py                # ResNet18+CBAM backbone, DQN, Dueling DQN, PPO
├── agents.py                  # DQN, Dueling DQN, PPO agent logic
├── ensemble.py                # Sortino-weighted voting ensemble
└── evaluate.py                # Evaluation metrics and visualization
```

## Data Format

### OHLCV CSV (`btc_1min.csv`):
```
timestamp,open,high,low,close,volume
1577836800000,7195.24,7196.25,7178.20,7180.97,123.45
...
```

### Funding Rate CSV (`funding_rate.csv`):
```
timestamp,funding_rate
1577836800000,0.0001
...
```

## Setup

```bash
pip install torch torchvision numpy pandas matplotlib pillow gym tqdm
```

## Usage

```bash
python main.py --ohlcv_path data/btc_1min.csv --funding_path data/funding_rate.csv
```

## Key Parameters (config.py)
- Trading interval: 15 minutes
- Image size: 224x224 RGB
- DRL algorithms: DQN, Dueling DQN, PPO
- Ensemble: Top 3 agents by Sortino ratio, weighted voting
- Walk-forward: 9 folders, 180-day train, 80-day validation, 80-day rolling window
- Transaction cost: 0.05%
- Initial balance: 100,000 USDT
