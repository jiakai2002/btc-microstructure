# Market Making Bot

A limit order market making simulation built on real Binance Futures L2 order book data.

## Overview

Streams live BTC/USDT order book data and simulates a market making strategy with volatility-adjusted spreads, inventory skewing, and order book imbalance-based quote adjustment.

## Strategy

Quotes a bid and ask around mid price every tick. Spread and quote placement are adjusted by three factors:

- **Volatility** — widens spread when market is volatile to compensate for adverse selection risk
- **Inventory skew** — shifts both quotes against the current position to mean-revert inventory toward zero
- **OBI signal** — nudges quotes in the direction order book imbalance predicts price will move

A one-step delay between placing and checking orders prevents quotes from filling against the same snapshot they were placed on.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # mac/linux
.venv\Scripts\activate     # windows
pip install -r requirements.txt
```

Collect data:
```bash
python stream.py
```

Run backtest:
```
jupyter notebook backtest.ipynb
```

## Key Results

| Metric | Value |
|---|---|
| Data | 5000 BTC/USDT L2 snapshots @ 100ms |
| OBI model accuracy | 95.2% (time series CV) |
| Base spread | 2 bps |
| Order size | 0.001 BTC |
| Max inventory | 0.1 BTC |

## Structure

```
orderbook.py      — OrderBookManager: maintains L2 order book from Binance WebSocket feed
stream.py         — streams live order book snapshots and saves to parquet
exchange.py       — simulated limit order exchange with partial fills and fee tracking
market_maker.py   — market making strategy
backtest.ipynb    — strategy backtest: PnL, inventory, fill analysis
```