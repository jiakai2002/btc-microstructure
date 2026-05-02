import math
import warnings
from collections import deque
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

warnings.simplefilter("ignore", OptimizeWarning)


class VolatilityEstimator:
    """
    EWMA volatility on log returns, scaled to horizon_sec.
    Fix 7: returns (sigma, vol_ratio) so caller can detect spikes.
    """

    def __init__(self, lambda_=0.97, floor=1e-6, cap=0.5,
                 horizon_sec=1.0, warmup=20):
        self.lambda_ = lambda_
        self.floor = floor
        self.cap = cap
        self.horizon_sec = horizon_sec
        self.warmup = warmup
        self.prev_mid: Optional[float] = None
        self.prev_ts: Optional[int] = None
        self.var_per_sec = floor * floor
        self.samples = 0
        self.last_sigma = floor
        self._prev_sigma = floor  # Fix 7: track prior sigma for spike ratio

    def update(self, mid: float, ts_ms: int) -> Tuple[float, float]:
        """Returns (sigma, vol_ratio). vol_ratio > spike_threshold → cancel quotes."""
        if mid <= 0:
            return self.last_sigma, 1.0
        if self.prev_mid is None:
            self.prev_mid = mid
            self.prev_ts = ts_ms
            return self.last_sigma, 1.0

        dt = (ts_ms - self.prev_ts) / 1000.0
        if dt <= 0:
            return self.last_sigma, 1.0

        r = math.log(mid / self.prev_mid)
        r = max(min(r, 0.05), -0.05)
        self.prev_mid = mid
        self.prev_ts = ts_ms

        inst_var = (r * r) / max(dt, 1e-3)
        self.var_per_sec = self.lambda_ * self.var_per_sec + (1.0 - self.lambda_) * inst_var

        sigma_log = math.sqrt(self.var_per_sec * self.horizon_sec)
        sigma_log = max(self.floor, min(self.cap, sigma_log))
        new_sigma = sigma_log * mid

        # Fix 7: compute spike ratio before updating
        vol_ratio = new_sigma / max(self._prev_sigma, self.floor)
        self._prev_sigma = new_sigma
        self.last_sigma = new_sigma
        self.samples += 1
        return self.last_sigma, vol_ratio

    @property
    def ready(self) -> bool:
        return self.samples >= self.warmup


class TradingIntensityIndicator:
    """
    Estimates κ and α by fitting λ(δ) = α·exp(−κ·δ) to live trade data.

    Fix 1: normalises aggregated volume by window duration → arrival rate proxy.
    Fix 2: timestamped mid-price buffer so price_level uses mid just before trade.
    """

    def __init__(self, sampling_length: int = 30, min_samples: int = 10,
                 smooth_alpha: float = 0.2, kappa_min: float = 0.05,
                 kappa_max: float = 50.0, mid_history_len: int = 50):
        self.sampling_length = sampling_length
        self.min_samples = min_samples
        self.smooth_alpha = smooth_alpha
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max

        # Fix 2: ring-buffer of (timestamp_ms, mid) snapshots
        self._mid_history: deque[Tuple[int, float]] = deque(maxlen=mid_history_len)

        self._current_sample: list[dict] = []
        self._samples: dict[int, list[dict]] = {}
        self.alpha: float = 1.0
        self.kappa: float = 1.5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_mid(self, best_bid: float, best_ask: float, timestamp_ms: int):
        """Call on every order-book tick. Fix 2: stores timestamped snapshot."""
        mid = 0.5 * (best_bid + best_ask)
        self._mid_history.append((timestamp_ms, mid))

    def on_trade(self, price: float, qty: float, timestamp_ms: int):
        """
        Call on every aggTrade event.
        Fix 2: resolves mid from the snapshot just before the trade,
        not the most-recently-seen mid.
        """
        mid = self._mid_at(timestamp_ms)
        if mid is None:
            return
        price_level = abs(price - mid)
        self._current_sample.append({"price_level": price_level, "amount": qty})

    def flush_sample(self, timestamp_ms: int):
        """Call periodically (e.g. every kappa_recalib_ticks). Runs fit when ready."""
        if not self._current_sample:
            return
        self._samples[timestamp_ms] = self._current_sample
        self._current_sample = []

        if len(self._samples) > self.sampling_length:
            oldest = min(self._samples.keys())
            del self._samples[oldest]

        if self.ready:
            self._fit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mid_at(self, timestamp_ms: int) -> Optional[float]:
        """Fix 2: last mid whose timestamp is strictly before the trade."""
        result = None
        for ts, mid in self._mid_history:
            if ts < timestamp_ms:
                result = mid
            else:
                break
        # fall back to most recent if nothing precedes the trade
        if result is None and self._mid_history:
            result = self._mid_history[-1][1]
        return result

    def _fit(self):
        # Aggregate volume per price level across all tick buckets
        trades_consolidated: dict[float, float] = {}
        for tick in self._samples.values():
            for t in tick:
                pl = t["price_level"]
                trades_consolidated[pl] = trades_consolidated.get(pl, 0) + t["amount"]

        # Fix 1: convert to arrival rate (volume / sec) so κ is window-size invariant
        timestamps = self._samples.keys()
        window_duration_sec = (max(timestamps) - min(timestamps)) / 1000.0

        price_levels = sorted(trades_consolidated.keys(), reverse=True)
        lambdas = [
            max(trades_consolidated[pl] / max(window_duration_sec, 1.0), 1e-10)
            for pl in price_levels
        ]

        try:
            params, _ = curve_fit(
                lambda t, a, b: a * np.exp(-b * t),
                price_levels,
                lambdas,
                p0=(self.alpha, self.kappa),
                method='dogbox',
                bounds=([0, 0], [np.inf, np.inf]),
            )
            k_new = float(np.clip(params[1], self.kappa_min, self.kappa_max))
            self.kappa = (1 - self.smooth_alpha) * self.kappa + self.smooth_alpha * k_new
            self.alpha = float(params[0])
        except (RuntimeError, ValueError):
            pass  # keep last valid estimate

    @property
    def ready(self) -> bool:
        return len(self._samples) >= self.min_samples
