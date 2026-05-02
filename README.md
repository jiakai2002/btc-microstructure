# Avellaneda-Stoikov Market Maker

A Python implementation of the Avellaneda-Stoikov (2008) optimal market-making framework, extended for continuous 24/7 crypto markets. Supports live trading on Binance USDT-M Futures and tick-level backtesting from streamed order book data.

---

## Overview

The strategy quotes a bid and ask around a **reservation price** that skews with inventory, with a **spread** that widens with volatility and narrows with liquidity. All parameters are estimated live from market data — no manual calibration required.

```
reservation price  r = mid − q · γ · σ² · τ
optimal half-spread  δ = ½ [ γ · σ² · τ + (2/γ) · ln(1 + γ/κ) ]
```

Key departures from the original paper:

| Paper assumption | This implementation |
|---|---|
| Finite trading horizon | Infinite horizon (τ=1), correct for 24/7 crypto |
| Fixed γ | Dynamic γ recomputed from spread bounds and inventory skew each cycle |
| Fixed κ | Live-calibrated via exponential curve fit to aggTrade depth data |
| Constant order size | Exponential decay with \|q\| (Fushimi et al. 2018) |
| Deterministic fills | Power queue fill model with configurable queue position assumption |

---

## Architecture

```
stream.py          — Binance USDT-M Futures order book WebSocket collector
orderbook.py       — Incremental order book with snapshot/delta reconciliation
indicators.py      — EWMA volatility estimator + TradingIntensityIndicator (κ, α)
strategy.py        — ASConfig, ASQuoter (reservation price, spread, quotes)
exchange.py        — Simulated exchange with power queue fill model
market_maker.py    — Orchestration: tick loop, order lifecycle, logging
logger.py          — Structured console logger
```

### Live data flow

```
Binance depth WS (100ms)  →  OrderBookManager  →  MarketMaker.on_tick()
Binance aggTrade WS        →  TradingIntensityIndicator.on_trade()
```

Both streams run as concurrent `asyncio` coroutines. The κ estimator consumes trades independently of the quote loop, ensuring no trade events are dropped during order book processing.

---

## Parameter Reference

### `ASConfig`

| Parameter | Default | Description |
|---|---|---|
| `gamma` | `0.1` | Risk aversion fallback (used when `dynamic_gamma=False`) |
| `kappa` | `1.5` | Order book depth factor — updated live by `TradingIntensityIndicator` |
| `infinite_horizon` | `True` | Sets τ=1; correct for continuous markets. Set `False` for vol-driven τ decay |
| `tau_decay` | `0.5` | Decay rate when `infinite_horizon=False`: τ = exp(−tau\_decay · max(vol\_ratio−1, 0)) |
| `dynamic_gamma` | `True` | Recompute γ each cycle from spread bounds and \|q\| |
| `inventory_risk_aversion` | `0.5` | IRA ∈ (0,1] — scales γ relative to its theoretical maximum |
| `gamma_cap` | `2.0` | Hard upper bound on dynamic γ |
| `min_spread` | `0.50` | Minimum half-spread in quote currency |
| `max_spread` | `20.0` | Maximum half-spread — used to bound dynamic γ |
| `vol_spike_threshold` | `2.0` | vol\_ratio above this triggers immediate quote cancel + κ recalibration |
| `max_inventory` | `0.05` | One-sided quoting beyond this threshold |
| `order_size` | `0.001` | Base order size in BTC |
| `eta_decay` | `0.0` | Size decay factor: size = base · exp(−η·\|q\|). 0 = constant size |
| `kappa_recalib_ticks` | `100` | Ticks between κ calibration windows |
| `kappa_sampling_length` | `30` | Rolling window size for κ estimation (tick buckets) |
| `kappa_min_samples` | `10` | Minimum samples before κ estimation begins |

---

## κ Estimation

The `TradingIntensityIndicator` estimates the order book depth parameter κ by fitting:

```
λ(δ) = α · exp(−κ · δ)
```

to live aggTrade data, where δ is the distance of each trade from the prevailing mid-price.

Implementation details:
- **Timestamped mid buffer** — each trade's δ is computed against the mid snapshot from just before the trade timestamp, not the current mid
- **Arrival rate normalization** — aggregated volume is divided by window duration (seconds) so κ is invariant to window size
- **EMA smoothing** — new estimates are blended: `κ ← (1−α)·κ_prev + α·κ_new`
- **Warm start** — previous (α, κ) used as initial guess for `scipy.optimize.curve_fit`
- **Fallback** — on fit failure, last valid estimate is retained

---

## Fill Model

The simulated exchange uses a **power queue fill model** (hftbacktest, 2023):

```
queue_ahead = fill_prob_scale × queue_size
base_prob   = remaining / max(queue_ahead, remaining)
prob        = 1 − (1 − base_prob)^power
```

`power=2` (default) produces a concave probability curve — orders near the front of the queue fill at disproportionately higher rates than back-of-queue orders, consistent with empirical data from Binance BTC perpetual (Albers et al. 2025).

| Parameter | Default | Effect |
|---|---|---|
| `fill_prob_scale` | `0.5` | Assumed queue position fraction. 0.3 = near front, 0.8 = near back |
| `power` | `2.0` | Curve concavity. Increase to 3 in more competitive markets |

Calibrate by comparing backtest fill rate against live fill rate and adjusting `power`.

---

## Quickstart

### Install

```bash
pip install -r requirements.txt
```

### Stream data

```bash
python stream.py          # streams BTCUSDT for 10 minutes → data/raw/book_<ts>.parquet
```

### Backtest

```bash
python market_maker.py backtest
```

If no parquet file exists at the configured path, the backtester will stream fresh data automatically.

### Live trading

```bash
python market_maker.py live
```

> **Note:** Live mode connects to Binance USDT-M Futures WebSocket feeds. No order submission is implemented — the live loop runs the strategy logic and logs quotes without placing real orders. Add exchange API credentials and order submission to `market_maker.py` to enable execution.

---

## Volatility Estimation

EWMA on log returns scaled to `vol_horizon_sec`:

```
σ²_t = λ · σ²_{t-1} + (1−λ) · r²_t / dt
σ = sqrt(σ² · horizon_sec) · mid
```

Log returns are clipped at ±5% to suppress data spikes. The estimator returns `(sigma, vol_ratio)` where `vol_ratio = σ_t / σ_{t-1}` — used to detect volatility spikes and trigger emergency quote cancellation.

---

## References

- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance, 8(3), 217–224.
- Cont, R., Stoikov, S. & Talreja, R. (2010). *A stochastic model for order book dynamics.* Operations Research, 58(3), 549–563.
- Fushimi, T., Gonzalez Garcia, C. & Herman, R. (2018). *Optimal High-Frequency Market Making.*
- Albers, J., Cucuringu, M., Howison, S. & Shestopaloff, A. (2025). *To Make, or To Take, That Is the Question.* arXiv:2502.18625.
- hftbacktest (2023). *Probability Queue Position Models.* https://hftbacktest.readthedocs.io
