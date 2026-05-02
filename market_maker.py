import asyncio
import math
import os
import time
from typing import Optional

import pandas as pd

from exchange import SimulatedExchange
from indicators import TradingIntensityIndicator, VolatilityEstimator
from strategy import ASConfig, ASQuoter
from logger import get_logger

logger = get_logger("market_maker")


class MarketMaker:
    def __init__(self, cfg: ASConfig = None):
        self.cfg = cfg or ASConfig()
        self.exchange = SimulatedExchange()
        self.quoter = ASQuoter(self.cfg)
        self._init_estimators()

        self.inventory: float = 0.0
        self.session_start: int = int(time.time() * 1000)
        self._session_count: int = 0          # Fix 5: track session wraps
        self._tick_count: int = 0
        self._calib_tick: int = 0

        self._active_bid_id: Optional[int] = None
        self._active_ask_id: Optional[int] = None
        self._active_bid_px: Optional[float] = None
        self._active_ask_px: Optional[float] = None
        self._active_half_spread: float = 0.0  # Fix 3: track last placed spread width

        self._log: list[dict] = []
        self._fill_log: list[dict] = []
        self._quote_log: list[dict] = []
        self._quotes_placed: int = 0
        self._fills_bid: int = 0
        self._fills_ask: int = 0

    # ------------------------------------------------------------------
    # Initialisation helpers (also used for Fix 5 session reset)
    # ------------------------------------------------------------------

    def _init_estimators(self):
        self.vol_est = VolatilityEstimator(
            horizon_sec=self.cfg.vol_horizon_sec,
            cap=self.cfg.vol_cap,
            floor=self.cfg.vol_floor,
        )
        self.kappa_calib = TradingIntensityIndicator(
            sampling_length=self.cfg.kappa_sampling_length,
            min_samples=self.cfg.kappa_min_samples,
        )

    # ------------------------------------------------------------------
    # Fix 5: session time with wrap detection
    # ------------------------------------------------------------------

    def _current_t(self, now_ms: int) -> float:
        elapsed_ms = now_ms - self.session_start
        session_ms = int(self.cfg.session_minutes * 60_000)
        current_session = elapsed_ms // session_ms

        if current_session > self._session_count:
            self._session_count = current_session
            self._on_session_reset()

        return (elapsed_ms % session_ms) / session_ms

    def _on_session_reset(self):
        """Fix 5: fresh estimators at the start of each new session."""
        logger.info("Session wrap — recalibrating estimators")
        self._init_estimators()

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------

    def _process_fills(self, fills) -> None:
        for fill in fills:
            if fill.side == "buy":
                self.inventory += fill.size
                self._fills_bid += 1
            else:
                self.inventory -= fill.size
                self._fills_ask += 1
            self._fill_log.append({
                "timestamp": fill.timestamp,
                "side":      fill.side,
                "price":     fill.price,
                "size":      fill.size,
                "fee":       fill.fee,
                "inventory": self.inventory,
            })
            logger.info(
                f"FILL {fill.side} {fill.size} BTC @ {fill.price:.2f} "
                f"fee=${fill.fee:.4f}"
            )

    def _quotes_stale(self, new_bid, new_ask, new_half: float) -> bool:
        """
        Fix 3: also triggers refresh when spread width changes by ≥ 1 tick,
        catching vol-spike repricing where mid hasn't moved.
        """
        tick = self.cfg.tick_size

        def px_moved(new, old):
            return (new is None) != (old is None) or (
                new is not None and old is not None and abs(new - old) >= tick
            )

        spread_changed = abs(new_half - self._active_half_spread) >= tick
        return px_moved(new_bid, self._active_bid_px) \
            or px_moved(new_ask, self._active_ask_px) \
            or spread_changed

    def _cancel_stale_quotes(self) -> None:
        if self._active_bid_id is not None:
            self.exchange.cancel_order(self._active_bid_id)
            self._active_bid_id = None
            self._active_bid_px = None
        if self._active_ask_id is not None:
            self.exchange.cancel_order(self._active_ask_id)
            self._active_ask_id = None
            self._active_ask_px = None

    def _place_quotes(self, bid_price, ask_price, half: float,
                      timestamp: int) -> None:
        tick = self.cfg.tick_size
        q = self.inventory - self.cfg.target_inventory
        size = self.quoter.order_size(q)  # Fix 6: size decays with |q|

        if bid_price is not None:
            bid_price = math.floor(bid_price / tick) * tick
            self._active_bid_id = self.exchange.place_limit_order(
                "buy", bid_price, size, timestamp)
            self._active_bid_px = bid_price
            self._quotes_placed += 1
            self._quote_log.append(
                {"timestamp": timestamp, "side": "bid",
                 "price": bid_price, "size": size})
            logger.info(f"ORDER buy {size:.6f} BTC @ {bid_price:.2f}")

        if ask_price is not None:
            ask_price = math.ceil(ask_price / tick) * tick
            self._active_ask_id = self.exchange.place_limit_order(
                "sell", ask_price, size, timestamp)
            self._active_ask_px = ask_price
            self._quotes_placed += 1
            self._quote_log.append(
                {"timestamp": timestamp, "side": "ask",
                 "price": ask_price, "size": size})
            logger.info(f"ORDER sell {size:.6f} BTC @ {ask_price:.2f}")

        self._active_half_spread = half  # Fix 3: record placed spread width

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def on_tick(self, row: dict) -> None:
        ts = row["timestamp"]
        mid = (row["bid_0_price"] + row["ask_0_price"]) / 2.0

        # Fix 2: pass timestamp so kappa_calib can match trades to correct mid
        self.kappa_calib.update_mid(row["bid_0_price"], row["ask_0_price"], ts)

        # Fix 7: unpack (sigma, vol_ratio)
        sigma, vol_ratio = self.vol_est.update(mid, ts)

        if not self.vol_est.ready:
            if self._tick_count % 10 == 0:
                logger.info(
                    f"warming up volatility... "
                    f"{self.vol_est.samples}/{self.vol_est.warmup}"
                )
            self._tick_count += 1
            return

        # Fix 7: volatility spike → cancel immediately, force recalibration
        if vol_ratio > self.cfg.vol_spike_threshold:
            logger.info(
                f"[vol spike] ratio={vol_ratio:.2f} — cancelling quotes"
            )
            self._cancel_stale_quotes()
            self.kappa_calib.flush_sample(ts)

        self._calib_tick += 1
        if self._calib_tick % self.cfg.kappa_recalib_ticks == 0:
            self.kappa_calib.flush_sample(ts)
            if self.kappa_calib.ready:
                self.cfg.kappa = self.kappa_calib.kappa
                logger.info(
                    f"[calib] κ={self.cfg.kappa:.4f}  "
                    f"α={self.kappa_calib.alpha:.6f} BTC/s"
                )

        t = self._current_t(ts)  # Fix 5: detects session wrap internally

        fills = self.exchange.check_fills(row)
        self._process_fills(fills)
        if fills:
            self._cancel_stale_quotes()

        self._tick_count += 1
        if self._tick_count % self.cfg.quote_refresh_ticks == 0:
            q = self.inventory - self.cfg.target_inventory
            bid_px, ask_px, half = self.quoter.quotes(mid, q, sigma, t)

            if self._quotes_stale(bid_px, ask_px, half):  # Fix 3
                self._cancel_stale_quotes()
                self._place_quotes(bid_px, ask_px, half, ts)

            if self._tick_count % 200 == 0:
                self._log_status(mid, bid_px, ask_px, half, sigma, q)

        self._log.append({
            "timestamp": ts,
            "mid":       mid,
            "inventory": self.inventory,
            "sigma":     sigma,
            "kappa":     self.cfg.kappa,
            "t":         t,
        })

    def _log_status(self, mid, bid_px, ask_px, half, sigma, q):
        tick = self.cfg.tick_size
        r = self.quoter.reservation_price(mid, q, sigma, self._current_t(
            int(time.time() * 1000)))
        pnl = self.exchange.realized_pnl(self.inventory, mid)
        bid_d = round(math.floor(bid_px / tick) * tick, 2) if bid_px else "N/A"
        ask_d = round(math.ceil(ask_px / tick) * tick, 2) if ask_px else "N/A"
        fill_rate = (
            (self._fills_bid + self._fills_ask) /
            max(self._quotes_placed, 1) * 100
        )
        gamma = self.quoter.effective_gamma(mid, q, sigma)
        size = self.quoter.order_size(q)
        logger.info(
            f"\n{'-'*38}\n"
            f"  mid price   = ${mid:.2f}\n"
            f"  bid / ask   = ${bid_d} / ${ask_d}\n"
            f"  half spread = ${half:.2f}\n"
            f"  volatility  = ${sigma:.4f}\n"
            f"  gamma (dyn) = {gamma:.4f}\n"
            f"  kappa       = {self.cfg.kappa:.3f}\n"
            f"  order size  = {size:.6f} BTC\n"
            f"  inventory   = {self.inventory:+.4f} BTC\n"
            f"  PnL         = ${pnl:+.4f}\n"
            f"  fills       = {self._fills_bid}b / {self._fills_ask}a\n"
            f"  fill rate   = {fill_rate:.1f}%\n"
            f"{'-'*38}\n"
        )

    # ------------------------------------------------------------------
    # Run interface
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> dict:
        logger.info(f"Running backtest on {len(df)} ticks…")
        for row in df.to_dict("records"):
            self.on_tick(row)
        self.exchange.cancel_all()
        final_mid = (
            df.iloc[-1]["bid_0_price"] + df.iloc[-1]["ask_0_price"]
        ) / 2.0
        summary = self.exchange.summary()
        summary["realized_pnl"] = round(
            self.exchange.realized_pnl(self.inventory, final_mid), 6)
        summary["final_inventory"] = round(self.inventory, 6)
        return summary

    def log_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._log)

    def fill_log_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._fill_log)

    def quote_log_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._quote_log)


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

async def run_live(symbol: str = "btcusdt", cfg: ASConfig = None):
    if cfg is None:
        cfg = ASConfig()

    from orderbook import OrderBookManager
    import websockets
    import json

    mm = MarketMaker(cfg)
    manager = OrderBookManager(symbol)
    logger.info(f"Running A-S market maker.  γ={cfg.gamma}  κ={cfg.kappa}")

    async def depth_loop():
        ws_url = f"wss://fstream.binance.com/ws/{symbol}@depth@100ms"
        async with websockets.connect(
            ws_url, ping_interval=20, ping_timeout=10
        ) as ws:
            for _ in range(10):
                manager.buffer.append(json.loads(await ws.recv()))
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, manager.fetch_snapshot)
            manager.apply_buffered_updates(manager.buffer)
            logger.info(f"Order book initialised for {symbol.upper()}")
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("u", 0) <= manager.last_update_id:
                    continue
                manager.apply_update(msg)
                row = manager.flatten()
                if row:
                    mm.on_tick(row)

    async def trade_loop():
        ws_url = f"wss://fstream.binance.com/market/ws/{symbol}@aggTrade"
        async with websockets.connect(
            ws_url, ping_interval=20, ping_timeout=10
        ) as ws:
            async for msg_raw in ws:
                msg = json.loads(msg_raw)
                # Fix 2: pass timestamp so mid buffer lookup is precise
                mm.kappa_calib.on_trade(
                    float(msg["p"]), float(msg["q"]), int(msg["T"])
                )

    await asyncio.gather(depth_loop(), trade_loop())


# ---------------------------------------------------------------------------
# Backtest mode
# ---------------------------------------------------------------------------

def run_backtest(parquet_path: str, cfg: ASConfig = None) -> dict:
    if cfg is None:
        cfg = ASConfig()

    if not os.path.exists(parquet_path):
        from stream import stream
        logger.info(f"No data at {parquet_path}, collecting...")
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        df = asyncio.run(stream("btcusdt", n_rows=10_000))
        df.to_parquet(parquet_path)
        logger.info(f"Saved {len(df)} rows → {parquet_path}")

    df = pd.read_parquet(parquet_path)
    mm = MarketMaker(cfg)
    summary = mm.run(df)
    logger.info("── Backtest Summary ──────────────────────")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
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
