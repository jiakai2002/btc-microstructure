import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from exchange import SimulatedExchange


@dataclass
class ASConfig:
    # A-S parameters
    gamma: float = 0.1          # inventory risk aversion  (γ)
    kappa: float = 1.5          # order-book depth factor  (κ) in 1/USD

    # Session
    session_minutes: float = 60.0   # T horizon for A-S time decay

    # Volatility estimation
    vol_horizon_sec: float = 1.0   # quoting horizon in seconds
    vol_cap: float = 0.5           # max σ in USD/sqrt(sec)
    vol_floor: float = 1e-6         # minimum σ to avoid div-by-zero

    # Inventory control
    max_inventory: float = 0.05     # max |q| in BTC before we stop quoting new
                                    # orders on the over-exposed side
    target_inventory: float = 0.0   # desired net position (usually 0)

    # Order sizing
    order_size: float = 0.001       # BTC per order
    min_spread: float = 0.50        # minimum half-spread in USD (floor)

    # Exchange
    fee_bps: float = 2.0            # taker fee in basis points
    tick_size: float = 0.10         # BTC/USDT tick size on Binance Futures

    # Kappa calibration
    # κ is re-estimated once per session reset from the aggTrade stream.
    # σ is re-estimated every tick via EWMA (no separate schedule needed).
    kappa_recalib_minutes: float = 5.0   # how often to refit κ
    kappa_calib_window: int  = 10_000  # number of trades to keep in rolling buffer
    kappa_calib_bins: int    = 20      # number of δ buckets for histogram fit
    kappa_min: float         = 0.05    # safety floor after calibration (1/USD)
    kappa_max: float         = 50.0    # safety ceiling after calibration (1/USD)

    # Misc
    quote_refresh_ticks: int = 1
    verbose: bool = True


class RollingVolatility:

    def __init__(
        self,
        lambda_: float = 0.97,
        floor: float = 1e-6,
        cap: float = 0.5,
        horizon_sec: float = 1.0,
        warmup: int = 30,
    ):
        self.lambda_     = lambda_
        self.floor       = floor
        self.cap         = cap
        self.horizon_sec = horizon_sec
        self.warmup      = warmup

        self.prev_mid    = None
        self.prev_ts     = None
        self.var_per_sec = floor * floor

        self.samples     = 0
        self.last_sigma  = floor
        self._recent     = deque(maxlen=100)

    def update(self, mid: float, ts_ms: int) -> float:
        """
        Call on each book update with latest midprice and timestamp.
        Returns current sigma scaled to quoting horizon.
        """
        if mid <= 0:
            return self.last_sigma

        if self.prev_mid is None:
            self.prev_mid = mid
            self.prev_ts  = ts_ms
            return self.last_sigma

        dt = (ts_ms - self.prev_ts) / 1000.0
        if dt <= 0:
            return self.last_sigma

        # log return
        r = math.log(mid / self.prev_mid)
        self.prev_mid = mid
        self.prev_ts  = ts_ms

        # optional clip for bad ticks
        r = max(min(r, 0.05), -0.05)

        # variance rate per sec
        inst_var = (r * r) / max(dt, 1e-3)

        # EWMA variance update
        self.var_per_sec = self.lambda_ * self.var_per_sec + (1.0 - self.lambda_) * inst_var

        # scale to quoting horizon (log-return volatility first)
        sigma_log = math.sqrt(self.var_per_sec * self.horizon_sec)
        sigma_log = max(self.floor, min(self.cap, sigma_log))

        sigma_usd = sigma_log * mid

        self.last_sigma = sigma_usd
        self.samples += 1
        self._recent.append(sigma_usd)

        return sigma_usd

    @property
    def ready(self) -> bool:
        return self.samples >= self.warmup

    @property
    def value(self) -> float:
        return self.last_sigma

    @property
    def mean_recent(self) -> float:
        if not self._recent:
            return self.floor
        return sum(self._recent) / len(self._recent)


class KappaCalibrator:
    """
    Rolling κ estimator for Avellaneda market making.

    Model:
        λ(δ) = A * exp(-κδ)
        log λ = log A - κδ

    Interpretation:
        Larger κ  -> fills decay quickly away from touch
        Smaller κ -> deeper quotes still trade
    """

    def __init__(
        self,
        tick_size: float,
        window: int = 5000,
        n_bins: int = 12,
        min_samples: int = 80,
        smooth_alpha: float = 0.15,
        kappa_min: float = 0.05,
        kappa_max: float = 50.0,
    ):
        self.tick_size = tick_size
        self.window = window
        self.n_bins = n_bins
        self.min_samples = min_samples
        self.alpha = smooth_alpha
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max

        self.mid = None

        # store delta_usd
        self.bid_hits = deque(maxlen=window)   # sell aggressor hits bids
        self.ask_hits = deque(maxlen=window)   # buy aggressor lifts asks

        self.k_bid = 1.5
        self.k_ask = 1.5

    def update_mid(self, best_bid: float, best_ask: float):
        self.mid = 0.5 * (best_bid + best_ask)

    def on_trade(self, price: float, qty: float, is_buyer_maker: bool):
        if self.mid is None:
            return
        delta_usd = abs(price - self.mid)
        if delta_usd <= 0:
            delta_usd = self.tick_size * 0.5

        if is_buyer_maker:
            self.bid_hits.append(delta_usd)
        else:
            self.ask_hits.append(delta_usd)

    def fit(self):
        kb = self._fit_side(self.bid_hits, self.k_bid)
        ka = self._fit_side(self.ask_hits, self.k_ask)

        self.k_bid = kb
        self.k_ask = ka

        return kb, ka

    def _fit_side(self, arr, prev_k):
        if len(arr) < self.min_samples:
            return prev_k

        x = np.array(arr, dtype=float)

        # trim top 5% of price distance
        hi = np.percentile(x, 95)
        x = x[x <= hi]
        if len(x) < self.min_samples:
            return prev_k

        # build histogram of price distance to count
        bins = np.linspace(0, max(self.tick_size, hi), self.n_bins + 1)
        counts, edges = np.histogram(x, bins=bins)
        mids = 0.5 * (edges[:-1] + edges[1:])
        mask = counts > 0
        if mask.sum() < 4:
            return prev_k
        d = mids[mask]   # price distances
        y = counts[mask].astype(float) # counts
        logy = np.log(y)

        # linear regression: log λ = a - κδ
        slope, intercept = np.polyfit(d, logy, 1)
        k_raw = -slope
        if not np.isfinite(k_raw):
            return prev_k

        k_raw = float(np.clip(k_raw, self.kappa_min, self.kappa_max))

        # smooth output
        k_new = (1 - self.alpha) * prev_k + self.alpha * k_raw
        return float(k_new)

    @property
    def ready(self):
        return (
            len(self.bid_hits) >= self.min_samples and
            len(self.ask_hits) >= self.min_samples
        )

    @property
    def value(self):
        return {
            "k_bid": self.k_bid,
            "k_ask": self.k_ask,
            "kappa": 0.5 * (self.k_bid + self.k_ask)
        }


class AvellanedaStoikovQuoter:
    """
    Computes optimal bid/ask quotes given the current market state.
    """

    def __init__(self, cfg: ASConfig):
        self.cfg = cfg

    def time_factor(self, t: float) -> float:
        """(T - t) clamped to [0, 1]."""
        return max(0.0, min(1.0, 1.0 - t))

    def reservation_price(self, mid: float, q: float, sigma: float, t: float) -> float:
        tau = self.time_factor(t)
        return mid - q * self.cfg.gamma * (sigma ** 2) * tau

    def optimal_spread(self, mid: float, sigma: float, t: float) -> float:
        tau   = self.time_factor(t)
        gamma = self.cfg.gamma
        kappa = self.cfg.kappa  # in 1/USD
        term1 = gamma * (sigma ** 2) * tau
        term2 = (2.0 / gamma) * math.log(1.0 + gamma / kappa)
        # fee floor: 2× one-way fee to break even on a round trip
        fee = (self.cfg.fee_bps / 10_000) * mid
        half_spread = max((term1 + term2) / 2.0, self.cfg.min_spread, 2.0 * fee)
        return half_spread

    def quotes(self, mid: float, q: float, sigma: float, t: float) -> tuple[Optional[float], Optional[float]]:
        """
        Returns (bid_price, ask_price).
        Returns None on the side we should not quote due to inventory limits.
        """
        r    = self.reservation_price(mid, q, sigma, t)
        half = self.optimal_spread(mid, sigma, t)
        bid_price = r - half
        ask_price = r + half
        max_inv = self.cfg.max_inventory

        # Suppress quoting on the side that would worsen inventory imbalance
        if q >= max_inv:
            bid_price = None   # already too long, don't buy more
        if q <= -max_inv:
            ask_price = None   # already too short, don't sell more

        return bid_price, ask_price


class MarketMaker:
    def __init__(self, cfg: ASConfig = ASConfig()):
        self.cfg      = cfg
        self.exchange = SimulatedExchange(fee_bps=cfg.fee_bps)
        self.quoter   = AvellanedaStoikovQuoter(cfg)
        self.vol_est = RollingVolatility(
            horizon_sec = cfg.vol_horizon_sec,
            cap         = cfg.vol_cap,
            floor       = cfg.vol_floor,
        )
        self.kappa_calib = KappaCalibrator(
            tick_size = cfg.tick_size,
            window    = cfg.kappa_calib_window,
            n_bins    = cfg.kappa_calib_bins,
            kappa_min = cfg.kappa_min,
            kappa_max = cfg.kappa_max,
        )

        # State
        self.inventory: float   = 0.0
        self.session_start: int = int(time.time() * 1000)
        self._tick_count: int   = 0
        self._active_bid_id: Optional[int] = None
        self._active_ask_id: Optional[int] = None
        self._active_bid_px: Optional[float] = None
        self._active_ask_px: Optional[float] = None
        self._initial_calib_done = False
        self._last_recalib_idx: int = -1

        # Logging
        self._log: list[dict] = []
        self._calib_log: list[dict] = []

    def _current_t(self, now_ms: int) -> tuple[float, int, int]:
        """
        fraction of current session elapsed [0, 1) and how many sessions have passed
        """
        elapsed_ms  = now_ms - self.session_start
        session_ms  = self.cfg.session_minutes * 60_000
        recalib_ms  = self.cfg.kappa_recalib_minutes * 60_000
        session_idx = int(elapsed_ms // session_ms)
        recalib_idx = int(elapsed_ms // recalib_ms)
        t = (elapsed_ms % session_ms) / session_ms
        return t, session_idx, recalib_idx

    def _maybe_recalibrate_kappa(self, recalib_idx: int, ts: int) -> None:
        """
        refit κ periodically using accumulated trades
        """
        if recalib_idx <= self._last_recalib_idx:
            return
        self._last_recalib_idx = recalib_idx

        if not self.kappa_calib.ready:
            if self.cfg.verbose:
                print(f"[calib] not enough data — keeping κ={self.cfg.kappa:.4f}")
            return

        k_bid, k_ask = self.kappa_calib.fit()
        kappa_new = float(np.clip(0.5 * (k_bid + k_ask), self.cfg.kappa_min, self.cfg.kappa_max))

        old = self.cfg.kappa
        self.cfg.kappa = kappa_new

        self._calib_log.append({"timestamp": ts, "k_bid": k_bid, "k_ask": k_ask, "kappa": kappa_new})
        if self.cfg.verbose:
            print(f"[calib] recalib {recalib_idx}  k_bid={k_bid:.3f}  k_ask={k_ask:.3f}  κ: {old:.4f} → {kappa_new:.4f} USD⁻¹")

    def _process_fills(self, fills) -> None:
        """
        Updates inventory on fills
        """
        for fill in fills:
            if fill.side == "buy":
                self.inventory += fill.size
            else:
                self.inventory -= fill.size

    def _quotes_stale(self, new_bid, new_ask) -> bool:
        tick = self.cfg.tick_size
        bid_moved = (new_bid is None) != (self._active_bid_px is None) or (
            new_bid is not None and self._active_bid_px is not None and
            abs(new_bid - self._active_bid_px) >= tick
        )
        ask_moved = (new_ask is None) != (self._active_ask_px is None) or (
            new_ask is not None and self._active_ask_px is not None and
            abs(new_ask - self._active_ask_px) >= tick
        )
        return bid_moved or ask_moved

    def _cancel_stale_quotes(self) -> None:
        """Cancel existing quotes before placing new ones."""
        if self._active_bid_id is not None:
            self.exchange.cancel_order(self._active_bid_id)
            self._active_bid_id = None
            self._active_bid_px = None
        if self._active_ask_id is not None:
            self.exchange.cancel_order(self._active_ask_id)
            self._active_ask_id = None
            self._active_ask_px = None

    def _place_quotes(self, bid_price, ask_price, timestamp: int) -> None:
        size = self.cfg.order_size
        tick = self.cfg.tick_size
        if bid_price is not None:
            bid_price = math.floor(bid_price / tick) * tick
            self._active_bid_id = self.exchange.place_limit_order("buy", bid_price, size, timestamp)
            self._active_bid_px = bid_price
        if ask_price is not None:
            ask_price = math.ceil(ask_price / tick) * tick
            self._active_ask_id = self.exchange.place_limit_order("sell", ask_price, size, timestamp)
            self._active_ask_px = ask_price

    def on_tick(self, row: dict) -> None:
        # main loop on every delta update
        ts  = row["timestamp"]
        mid = (row["bid_0_price"] + row["ask_0_price"]) / 2.0

        # 1. update mid for calibrator
        self.kappa_calib.update_mid(row["bid_0_price"], row["ask_0_price"])

        # 2. update vol via EWMA
        sigma = self.vol_est.update(mid, ts)
        if not self.vol_est.ready:
            return
        
        # 3. initial κ calibration as soon as enough trades accumulate
        if not self._initial_calib_done and self.kappa_calib.ready:
            k_bid, k_ask = self.kappa_calib.fit()
            self.cfg.kappa = float(np.clip(0.5 * (k_bid + k_ask), self.cfg.kappa_min, self.cfg.kappa_max))
            self._initial_calib_done = True
            if self.cfg.verbose:
                print(f"[calib] initial fit  κ={self.cfg.kappa:.4f} USD⁻¹")

        # 4. recalibrate κ periodically
        t, session_idx, recalib_idx = self._current_t(ts)
        self._maybe_recalibrate_kappa(recalib_idx, ts)

        # 5. check for fills on open orders
        fills = self.exchange.check_fills(row)
        self._process_fills(fills)
        if fills:
            self._cancel_stale_quotes()

        # 6. place new quotes
        self._tick_count += 1
        if self._tick_count % self.cfg.quote_refresh_ticks == 0:
            q = self.inventory - self.cfg.target_inventory
            bid_px, ask_px = self.quoter.quotes(mid, q, sigma, t)

            # only requote if stale 
            if self._quotes_stale(bid_px, ask_px):
                self._cancel_stale_quotes()
                self._place_quotes(bid_px, ask_px, ts)

            if self.cfg.verbose and self._tick_count % 500 == 0:
                r = self.quoter.reservation_price(mid, q, sigma, t)
                half = self.quoter.optimal_spread(mid, sigma, t)
                pnl  = self.exchange.realized_pnl(self.inventory, mid)
                print(
                    f"[{self._tick_count:>7}]  mid={mid:.2f}  r={r:.2f}  "
                    f"bid={bid_px or 'N/A'}  ask={ask_px or 'N/A'}  "
                    f"half=${half:.2f}  σ=${sigma:.4f}  κ={self.cfg.kappa:.3f}USD⁻¹  "
                    f"q={self.inventory:+.4f}BTC  pnl=${pnl:+.4f}"
                )

        # 7. Log
        self._log.append({
            "timestamp": ts,
            "mid":       mid,
            "inventory": self.inventory,
            "sigma":     sigma,
            "kappa":     self.cfg.kappa,
            "t":         t,
        })

    def run(self, df: pd.DataFrame) -> dict:
        """Feed a pre-collected DataFrame of order-book rows through the strategy."""
        print(f"Running A-S market maker on {len(df)} ticks…")
        for row in df.to_dict("records"):
            self.on_tick(row)

        self.exchange.cancel_all()
        final_mid = (df.iloc[-1]["bid_0_price"] + df.iloc[-1]["ask_0_price"]) / 2.0
        summary   = self.exchange.summary()
        summary["realized_pnl"] = round(self.exchange.realized_pnl(self.inventory, final_mid), 6)
        summary["final_inventory"] = round(self.inventory, 6)
        return summary

    def log_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._log)

    def calib_log_df(self) -> pd.DataFrame:
        """Returns a DataFrame of every κ recalibration event."""
        return pd.DataFrame(self._calib_log)


async def run_live(symbol: str = "btcusdt", cfg: ASConfig = None):
    """
    Runs two concurrent WebSocket streams:
      - @depth@100ms  → order book ticks → quotes
      - @aggTrade     → real market orders → κ calibration
    """
    if cfg is None:
        cfg = ASConfig()

    from orderbook import OrderBookManager
    import websockets, json

    mm = MarketMaker(cfg)
    manager = OrderBookManager(symbol)

    print(f"Live A-S market maker on {symbol.upper()}  γ={cfg.gamma}  κ={cfg.kappa}USD⁻¹")

    async def depth_loop():
        ws_url = f"wss://fstream.binance.com/ws/{symbol}@depth@100ms"
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
            # buffer, fetch snapshot, sync book
            for _ in range(10):
                manager.buffer.append(json.loads(await ws.recv()))
            manager.fetch_snapshot()
            manager.apply_buffered_updates(manager.buffer)
            print("Order book ready. Market making started.")
            while True:
                # process delta, flatten top 10, tick mm
                msg = json.loads(await ws.recv())
                if msg.get("u", 0) <= manager.last_update_id:
                    continue
                manager.apply_update(msg)
                row = manager.flatten()
                if row:
                    mm.on_tick(row)

    async def trade_loop():
        # feed trade msg into calibrator
        ws_url = f"wss://fstream.binance.com/ws/{symbol}@aggTrade"
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
            while True:
                msg = json.loads(await ws.recv())
                price = float(msg["p"])
                qty   = float(msg["q"])
                is_buyer_maker = msg["m"]
                mm.kappa_calib.on_trade(price, qty, is_buyer_maker)

    await asyncio.gather(depth_loop(), trade_loop())


def run_backtest(parquet_path: str, cfg: ASConfig = None) -> dict:
    if cfg is None:
        cfg = ASConfig()
    df      = pd.read_parquet(parquet_path)
    mm      = MarketMaker(cfg)
    summary = mm.run(df)

    print("\n── Backtest Summary ──────────────────────")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    calib = mm.calib_log_df()
    if not calib.empty:
        print(f"\n── κ Calibration History ({len(calib)} sessions) ───")
        print(calib.to_string(index=False))

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["live", "backtest"])
    args = parser.parse_args()

    cfg = ASConfig()

    if args.mode == "live":
        asyncio.run(run_live("btcusdt", cfg))
    else:
        run_backtest("data/raw/book_1777480982.parquet", cfg)