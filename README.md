# Market Making Bot

A limit order market making simulation built on real Binance Futures L2 order book data.

## Overview

Streams live BTC/USDT order book data and simulates an Avellaneda-Stoikov market making strategy with an order book imbalance signal overlay. Quotes are placed around a risk-adjusted reservation price derived from inventory and volatility, rather than symmetrically around mid.

## Structure

```
orderbook.py      — OrderBookManager: maintains L2 order book from Binance WebSocket feed
stream.py         — streams live order book snapshots and saves to parquet
exchange.py       — simulated limit order exchange with partial fills and fee tracking
market_maker.py   — Avellaneda-Stoikov market making strategy with OBI overlay
backtest.ipynb    — strategy backtest: PnL, inventory, reservation price, fill analysis
```

## Strategy

Implements the Avellaneda-Stoikov (2008) market making model:

```
reservation_price = mid - q * gamma * sigma²
spread            = gamma * sigma² + (2/gamma) * ln(1 + gamma/kappa)
```

- **Reservation price** shifts away from mid based on inventory — long position lowers it to favour selling, short position raises it to favour buying
- **Spread** widens automatically with volatility and narrows with dense order flow (kappa)
- **OBI signal** nudges quotes in the direction order book imbalance predicts price will move
- **One-step delay** between placing and checking orders prevents quotes filling against the same snapshot they were placed on

## Quickstart

```bash
pip install -r requirements.txt
```

Collect data:
```bash
python stream.py
```

Run backtest:
```bash
jupyter notebook backtest.ipynb
```

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `gamma` | 0.1 | Risk aversion — higher widens spread and skews more aggressively |
| `kappa` | 1.5 | Order arrival rate — higher tightens spread |
| `order_size` | 0.001 BTC | Size per order |
| `max_inventory` | 0.1 BTC | Hard inventory limit |

## Key Results

| Metric | Value |
|---|---|
| Asset | BTC/USDT Perpetual Futures |
| Data | L2 order book snapshots @ 500ms |
| Model | Avellaneda-Stoikov (2008) |
