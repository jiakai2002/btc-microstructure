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
    # ── Core A-S parameters ──────────────────────────────────────────────
    gamma: float = 0.1          # inventory risk aversion  (γ)
    kappa: float = 1.5          # order-book depth factor  (κ)

    # ── Session / time ────────────────────────────────────────────────────
    session_minutes: float = 60.0   # T in wall-clock minutes; resets cyclically

    # ── Volatility estimation ─────────────────────────────────────────────
    vol_window: int = 100           # rolling window for σ (number of ticks)
    vol_floor: float = 1e-4         # minimum σ to avoid div-by-zero

    # ── Inventory control ─────────────────────────────────────────────────
    max_inventory: float = 0.05     # max |q| in BTC before we stop quoting new
                                    # orders on the over-exposed side
    target_inventory: float = 0.0   # desired net position (usually 0)

    # ── Order sizing ──────────────────────────────────────────────────────
    order_size: float = 0.001       # BTC per order
    min_spread: float = 0.50        # minimum half-spread in USD (floor)

    # ── Exchange ──────────────────────────────────────────────────────────
    fee_bps: float = 2.0            # taker fee in basis points

    # ── Calibration ───────────────────────────────────────────────────────
    # κ is re-estimated once per session reset from the aggTrade stream.
    # σ is re-estimated every tick via EWMA (no separate schedule needed).
    kappa_calib_window: int  = 10_000  # number of trades to keep in rolling buffer
    kappa_calib_bins: int    = 20      # number of δ buckets for histogram fit
    kappa_min: float         = 0.05    # safety floor after calibration
    kappa_max: float         = 50.0    # safety ceiling after calibration

    # ── Misc ──────────────────────────────────────────────────────────────
    quote_refresh_ticks: int = 1    # re-quote every N ticks (1 = every tick)
    verbose: bool = True


class RollingVolatility:
    """
    Exponentially-weighted σ of mid-price absolute increments.

    Why EWMA over plain std:
      - Reacts faster to volatility spikes (regime changes, news)
      - Decays stale observations smoothly instead of a hard window drop-off
      - Still uses a deque so memory is bounded by `window`

    σ is kept in raw USD-per-tick units to match the A-S Brownian-motion
    assumption (dS = σ dW over one 100 ms step).
    """

    def __init__(self, window: int = 100, floor: float = 1e-4):
        self.window = window
        self.floor  = floor
        self._mids: deque = deque(maxlen=window + 1)

    def update(self, mid: float) -> float:
        self._mids.append(mid)
        if len(self._mids) < 2:
            return self.floor
        mids    = np.array(self._mids)
        returns = np.diff(mids)
        # Exponential weights: oldest ≈ e^{-1}, newest = 1
        weights = np.exp(np.linspace(-1.0, 0.0, len(returns)))
        weights /= weights.sum()
        ewma_var = float(np.sum(weights * returns ** 2))
        return max(math.sqrt(ewma_var), self.floor)

    @property
    def ready(self) -> bool:
        return len(self._mids) > 10


class KappaCalibrator:
    """
    Estimates κ from the Binance aggTrade stream (same approach as Hummingbot).

    Theory (A-S §2.4):
        Market orders arrive at Poisson rate λ(δ) = A · exp(−κ · δ)
        where δ is the trade's distance from mid at the time it executed.
        κ controls how steeply arrival rate decays with distance.

    Method:
        Every aggTrade gives us a real market order: price p and quantity q.
        We record δ = |p - mid| for each trade (weighted by quantity so that
        large trades count more, matching the intensity model).
        At each session reset we bin trades by δ and fit the exponential to
        the bin counts — finding the κ that best matches the observed decay.
    """

    def __init__(self, window: int = 10_000, n_bins: int = 20,
                 kappa_min: float = 0.05, kappa_max: float = 50.0):
        self.n_bins    = n_bins
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        # Store (delta, qty) per trade; bounded rolling window
        self._buf: deque = deque(maxlen=window)
        self._current_mid: float = 0.0   # updated by on_tick

    def update_mid(self, mid: float) -> None:
        self._current_mid = mid

    def on_trade(self, price: float, qty: float) -> None:
        """Call for every aggTrade message received."""
        if self._current_mid <= 0:
            return
        delta = abs(price - self._current_mid)
        if delta > 0:
            self._buf.append((delta, qty))

    def fit(self) -> Optional[float]:
        """
        Bin trades by δ, then fit A·exp(−κδ) to (bin_midpoint, count).
        Returns κ, or None if insufficient data or fit fails.
        """
        if len(self._buf) < 50:
            return None

        data   = np.array(self._buf)
        deltas = data[:, 0]
        qtys   = data[:, 1]

        # Bin by delta; use qty-weighted counts so large trades count more
        delta_max = np.percentile(deltas, 95)   # ignore extreme outliers
        if delta_max <= 0:
            return None

        bins       = np.linspace(0, delta_max, self.n_bins + 1)
        bin_mids   = (bins[:-1] + bins[1:]) / 2
        counts, _  = np.histogram(deltas, bins=bins, weights=qtys)

        # Only fit bins with nonzero counts
        mask     = counts > 0
        if mask.sum() < 5:
            return None
        x = bin_mids[mask]
        y = counts[mask] / counts[mask].max()   # normalise to [0,1]

        try:
            (_, kappa_fit), _ = curve_fit(
                lambda d, A, k: A * np.exp(-k * d),
                x, y,
                p0=[1.0, 1.0],
                bounds=([0.0, self.kappa_min], [10.0, self.kappa_max]),
                maxfev=5_000,
            )
            return float(np.clip(kappa_fit, self.kappa_min, self.kappa_max))
        except (RuntimeError, ValueError):
            return None


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

    def optimal_spread(self, sigma: float, t: float) -> float:
        tau   = self.time_factor(t)
        gamma = self.cfg.gamma
        kappa = self.cfg.kappa
        term1 = gamma * (sigma ** 2) * tau
        term2 = (2.0 / gamma) * math.log(1.0 + gamma / kappa)
        full_spread = term1 + term2
        half_spread = max(full_spread / 2.0, self.cfg.min_spread)
        return half_spread

    def quotes(
        self,
        mid: float,
        q: float,
        sigma: float,
        t: float,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Returns (bid_price, ask_price).
        Returns None on the side we should not quote due to inventory limits.
        """
        r    = self.reservation_price(mid, q, sigma, t)
        half = self.optimal_spread(sigma, t)

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
        self.vol_est  = RollingVolatility(window=cfg.vol_window, floor=cfg.vol_floor)
        self.kappa_calib = KappaCalibrator(
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
        self._last_session_idx: int = -1

        # Logging
        self._log: list[dict] = []
        self._calib_log: list[dict] = []

    # ── Time helpers ─────────────────────────────────────────────────────

    def _current_t(self, now_ms: int) -> tuple[float, int]:
        """
        Returns (t, session_idx) where t ∈ [0,1) is progress through the
        current session and session_idx increments each time T resets.
        """
        elapsed_ms  = now_ms - self.session_start
        session_ms  = self.cfg.session_minutes * 60_000
        session_idx = int(elapsed_ms // session_ms)
        t = (elapsed_ms % session_ms) / session_ms
        return t, session_idx

    # ── κ recalibration ───────────────────────────────────────────────────

    def _maybe_recalibrate_kappa(self, session_idx: int, ts: int) -> None:
        """
        Fit κ from accumulated book data at the start of each new session.
        The calibrator's rolling buffer always holds the most recent
        `kappa_calib_window` ticks, so the fit reflects current regime.
        """
        if session_idx <= self._last_session_idx:
            return   # already calibrated this session

        kappa_new = self.kappa_calib.fit()
        self._last_session_idx = session_idx

        if kappa_new is None:
            if self.cfg.verbose:
                print(f"[calib] κ fit failed (not enough data) — keeping κ={self.cfg.kappa:.4f}")
            return

        old_kappa         = self.cfg.kappa
        self.cfg.kappa    = kappa_new
        self.quoter.cfg   = self.cfg   # quoter holds a reference, update in place

        self._calib_log.append({"timestamp": ts, "kappa": kappa_new})

        if self.cfg.verbose:
            print(
                f"[calib] session {session_idx}  "
                f"κ: {old_kappa:.4f} → {kappa_new:.4f}"
            )

    # ── Core per-tick logic ───────────────────────────────────────────────

    def _process_fills(self, fills) -> None:
        for fill in fills:
            if fill.side == "buy":
                self.inventory += fill.size
            else:
                self.inventory -= fill.size

    def _cancel_stale_quotes(self) -> None:
        """Cancel existing quotes before placing new ones."""
        if self._active_bid_id is not None:
            self.exchange.cancel_order(self._active_bid_id)
            self._active_bid_id = None
        if self._active_ask_id is not None:
            self.exchange.cancel_order(self._active_ask_id)
            self._active_ask_id = None

    def _place_quotes(self, bid_price, ask_price, timestamp: int) -> None:
        size = self.cfg.order_size
        if bid_price is not None:
            self._active_bid_id = self.exchange.place_limit_order("buy", bid_price, size, timestamp)
        if ask_price is not None:
            self._active_ask_id = self.exchange.place_limit_order("sell", ask_price, size, timestamp)

    def on_tick(self, row: dict) -> None:
        ts  = row["timestamp"]
        mid = (row["bid_0_price"] + row["ask_0_price"]) / 2.0

        # 1. Update mid for calibrator and compute EWMA volatility
        self.kappa_calib.update_mid(mid)
        sigma = self.vol_est.update(mid)
        if not self.vol_est.ready:
            return

        # 2. Check session boundary → recalibrate κ if a new session started
        t, session_idx = self._current_t(ts)
        self._maybe_recalibrate_kappa(session_idx, ts)

        # 3. Check for fills on open orders
        fills = self.exchange.check_fills(row)
        self._process_fills(fills)
        if fills:
            self._cancel_stale_quotes()

        # 4. Re-quote on schedule
        self._tick_count += 1
        if self._tick_count % self.cfg.quote_refresh_ticks == 0:
            self._cancel_stale_quotes()

            q              = self.inventory - self.cfg.target_inventory
            bid_px, ask_px = self.quoter.quotes(mid, q, sigma, t)

            self._place_quotes(bid_px, ask_px, ts)

            if self.cfg.verbose and self._tick_count % 500 == 0:
                r    = self.quoter.reservation_price(mid, q, sigma, t)
                half = self.quoter.optimal_spread(sigma, t)
                pnl  = self.exchange.realized_pnl(self.inventory, mid)
                print(
                    f"[{self._tick_count:>7}]  mid={mid:.2f}  r={r:.2f}  "
                    f"bid={bid_px or 'N/A'}  ask={ask_px or 'N/A'}  "
                    f"half_spread={half:.2f}  σ={sigma:.4f}  κ={self.cfg.kappa:.3f}  "
                    f"q={self.inventory:+.4f}  pnl={pnl:+.4f}"
                )

        # 5. Log
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

    print(f"Live A-S market maker on {symbol.upper()}  γ={cfg.gamma}  κ={cfg.kappa}")

    async def depth_loop():
        ws_url = f"wss://fstream.binance.com/ws/{symbol}@depth@100ms"
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
            for _ in range(10):
                manager.buffer.append(json.loads(await ws.recv()))
            manager.fetch_snapshot()
            manager.apply_buffered_updates(manager.buffer)
            print("Order book ready. Market making started.")
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("u", 0) <= manager.last_update_id:
                    continue
                manager.apply_update(msg)
                row = manager.flatten()
                if row:
                    mm.on_tick(row)

    async def trade_loop():
        ws_url = f"wss://fstream.binance.com/ws/{symbol}@aggTrade"
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
            while True:
                msg = json.loads(await ws.recv())
                # m=True means the buyer is the market maker → seller is aggressor
                price = float(msg["p"])
                qty   = float(msg["q"])
                mm.kappa_calib.on_trade(price, qty)

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

    cfg = ASConfig(
        gamma               = 0.1,
        kappa               = 1.5,       # initial κ before first calibration
        session_minutes     = 5.0,
        vol_window          = 100,
        order_size          = 0.001,
        max_inventory       = 0.05,
        min_spread          = 0.50,
        fee_bps             = 2.0,
        kappa_calib_window  = 10_000,    # trades to keep in rolling buffer
        kappa_calib_bins    = 20,        # δ buckets for histogram fit
        kappa_min           = 0.05,
        kappa_max           = 50.0,
        quote_refresh_ticks = 1,
        verbose             = True,
    )

    if args.mode == "live":
        asyncio.run(run_live("btcusdt", cfg))
    else:
        run_backtest("data/raw/book_1777480982.parquet", cfg)
