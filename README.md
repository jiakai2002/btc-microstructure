# BTC/USDT Market Maker

Inspired by the Avellaneda-Stoikov (2008) model. Supports live trading and backtesting on BTC/USDT perpetual futures from Binance.

<img width="575" height="562" alt="Screenshot 2026-05-03 at 2 55 19 AM" src="https://github.com/user-attachments/assets/f28b205b-0263-41a6-b26f-cf179de82e93" />

---

## Strategy

We quote a bid and ask around a **reservation price** that adjusts with inventory, with a **spread** that widens with volatility and narrows with liquidity.

```
r = mid − q · γ · σ² · τ
δ = ½ [ γ · σ² · τ  +  (2/γ) · ln(1 + γ/κ) ]
```

---

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

---

## Live Data Flow

```
Binance depth WS @100ms  →  OrderBookManager  →  MarketMaker.on_tick()
Binance aggTrade WS       →  TradingIntensityIndicator.on_trade()
```

Both streams run as concurrent coroutines — trade events feed the κ estimator independently of the quote loop.

---

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

---

## κ Estimation

Fits `λ(δ) = α · exp(−κ · δ)` to live aggTrade data, where δ is each trade's distance from mid at the time of the trade.

- **Timestamped mid buffer** — δ resolved against mid just before the trade, not current mid
- **Arrival rate normalisation** — volume divided by window duration so κ is window-size invariant
- **EMA smoothing** — `κ ← (1−α)·κ_prev + α·κ_new` with hard clamp `[kappa_min, kappa_max]`
- **Fallback** — last valid estimate retained on fit failure

---

## Volatility

EWMA on log returns, output in absolute price units:

```
σ²_t = λ · σ²_{t-1} + (1−λ) · r²_t / dt
σ    = sqrt(σ² · horizon_sec) · mid
```

Returns `(sigma, vol_ratio)`. When `vol_ratio > vol_spike_threshold`, quotes are cancelled and κ recalibration is forced immediately.

---

## Run

```bash
pip install -r requirements.txt

python market_maker.py backtest   # streams fresh data if parquet not found
python market_maker.py live       # logs quotes, no real order submission
```

---

## References

- Avellaneda & Stoikov (2008). *High-frequency trading in a limit order book.* Quantitative Finance 8(3).
- Cont, Stoikov & Talreja (2010). *A stochastic model for order book dynamics.* Operations Research 58(3).
- Fushimi, Gonzalez Garcia & Herman (2018). *Optimal High-Frequency Market Making.*
- Albers et al. (2025). *To Make, or To Take.* arXiv:2502.18625.
