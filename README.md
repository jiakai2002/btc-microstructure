# market making bot

Inspired by the Avellaneda-Stoikov (2008) model. Supports live trading and backtesting on BTC/USDT from Binance.

<img width="837" height="673" alt="Screenshot 2026-05-03 at 12 47 42 PM" src="https://github.com/user-attachments/assets/e6ba8663-55c5-4217-84a5-9acdac557e9c" />

## Strategy

We quote a bid and ask around a **reservation price** that adjusts with inventory, with a **spread** that widens with volatility and narrows with liquidity.
```
r = mid − q · γ · σ² · τ
δ = ½ [ γ · σ² · τ  +  (2/γ) · ln(1 + γ/κ) ]
```

## Files

| File | Description |
|---|---|
| `strategy.py` | Parameters, reservation price, optimal spread, quotes |
| `indicators.py` | EWMA volatility estimator + κ calibration via curve fit |
| `exchange.py` | Open orders, price-cross fill simulation, PnL |
| `market_maker.py` | Tick loop, order lifecycle, κ recalibration, logging |
| `orderbook.py` | Live order book — REST snapshot + WebSocket delta updates |
| `stream.py` | Streams Binance order book depth to parquet for backtesting |
| `logger.py` | Console logger |

## Live Data Flow

```
Binance depth WS  →  OrderBookManager  →  MarketMaker.on_tick()
Binance aggTrade WS  →  TradingIntensityIndicator.on_trade()
```

Both streams run as concurrent coroutines — trade events feed the κ estimator independently of the quote loop.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `gamma` | `0.1` | Risk aversion — fallback when `dynamic_gamma=False` |
| `kappa` | `1.5` | Order book depth — updated live by `TradingIntensityIndicator` |
| `infinite_horizon` | `True` | τ=1 always; set `False` for vol-driven τ decay |
| `tau_decay` | `0.5` | τ decay sensitivity when `infinite_horizon=False` |
| `dynamic_gamma` | `True` | Recompute γ each cycle from spread bounds and \|q\| |
| `inventory_risk_aversion` | `0.5` | Scales γ relative to its theoretical maximum |
| `gamma_cap` | `2.0` | Hard upper bound on dynamic γ |
| `min_spread` | `0.50` | Minimum half-spread (USD) |
| `max_spread` | `20.0` | Maximum half-spread — bounds dynamic γ |
| `vol_spike_threshold` | `2.0` | vol_ratio above this cancels quotes and forces κ recalibration |
| `max_inventory` | `0.05` | One-sided quoting beyond this threshold |
| `order_size` | `0.001` | Base order size in BTC |
| `eta_decay` | `0.0` | Size decay: `size = base · exp(−η·\|q\|)`. 0 = constant |
| `kappa_recalib_ticks` | `100` | Ticks between κ calibration windows |
| `kappa_sampling_length` | `30` | Rolling window size for κ estimation |
| `kappa_min_samples` | `10` | Minimum samples before κ estimation begins |

## Volatility
```
mid (each tick)
    |
    | r  = clip(log(mid / prev_mid), ±0.05)
    | dt = ms delta -> seconds
    v
    | inst_var   = r² / dt
    | var_per_sec = 0.97·var_prev + 0.03·inst_var     # EWMA
    v
    | sigma_log = clip(sqrt(var_per_sec · horizon), floor, cap)
    | sigma     = sigma_log · mid                      # -> USD
    | vol_ratio = sigma / prev_sigma
    v
(sigma, vol_ratio)
    |
    |-- sigma ------->  A-S reservation price + spread
    '-- vol_ratio -->  > threshold?  cancel quotes + force recalib
```

## κ Estimation
```
# two async streams run concurrently

aggTrade WS                          depth WS
    |                                    |
    | price, qty, timestamp_ms           | best_bid, best_ask, timestamp_ms
    v                                    v
on_trade()                          update_mid()
    |                                    |
    |   .-- lookup mid at trade time ----|
    |  /      from _mid_history deque <--'
    | /
    | delta = |price - mid|
    v
_current_sample  <-- accumulates {price_level, amount} per trade
    |
    | flush_sample()  <-- called every kappa_recalib_ticks
    |                     seals bucket, evicts oldest if > 30
    v
_samples dict  [t0: [...], t1: [...], ..., tN: [...]]
    |                rolling window, max 30 buckets
    | when len(_samples) >= min_samples
    v
_fit()
    |
    | 1. aggregate volume by price level across all buckets
    | 2. normalise by window duration  ->  lambda (BTC/s)
    | 3. curve_fit:  lambda(delta) = alpha * exp(-kappa * delta)
    | 4. clip kappa to [0.05, 50]
    | 5. EMA blend:  kappa = 0.8 * kappa_prev + 0.2 * kappa_new
    |    (on fit failure: retain last valid kappa)
    v
self.kappa  ->  ASConfig.kappa  ->  A-S spread formula
```

## Run

```bash
pip install -r requirements.txt

python market_maker.py backtest   # streams fresh data if parquet not found
python market_maker.py live       # logs quotes, no real order submission
```
