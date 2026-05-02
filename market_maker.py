import asyncio
import math
import os
from typing import Optional

import pandas as pd

from exchange import Exchange
from indicators import TradingIntensityIndicator, VolatilityEstimator
from strategy import ASConfig, ASQuoter
from logger import get_logger

logger = get_logger("market_maker")


class MarketMaker:
    def __init__(self, cfg: ASConfig = None):
        self.cfg = cfg or ASConfig()
        self.exchange = Exchange()
        self.quoter = ASQuoter(self.cfg)
        self._init_estimators()

        self.inventory: float = 0.0
        self._tick_count: int = 0
        self._calib_tick: int = 0

        self._active_bid_id: Optional[int] = None
        self._active_ask_id: Optional[int] = None
        self._active_bid_px: Optional[float] = None
        self._active_ask_px: Optional[float] = None
        self._active_half_spread: float = 0.0

        self._log: list[dict] = []
        self._fill_log: list[dict] = []
        self._quote_log: list[dict] = []
        self._quotes_placed: int = 0
        self._fills_bid: int = 0
        self._fills_ask: int = 0

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
            logger.info(f"FILL {fill.side} {fill.size} BTC @ {fill.price:.2f} fee=${fill.fee:.4f}")

    def _quotes_stale(self, new_bid, new_ask, new_half: float) -> bool:
        """Refresh if price moved ≥ 1 tick or spread width changed ≥ 1 tick."""
        tick = self.cfg.tick_size

        def px_moved(new, old):
            return (new is None) != (old is None) or (
                new is not None and old is not None and abs(new - old) >= tick
            )

        return (px_moved(new_bid, self._active_bid_px)
                or px_moved(new_ask, self._active_ask_px)
                or abs(new_half - self._active_half_spread) >= tick)

    def _cancel_stale_quotes(self) -> None:
        if self._active_bid_id is not None:
            self.exchange.cancel_order(self._active_bid_id)
            self._active_bid_id = None
            self._active_bid_px = None
        if self._active_ask_id is not None:
            self.exchange.cancel_order(self._active_ask_id)
            self._active_ask_id = None
            self._active_ask_px = None

    def _place_quotes(self, bid_price, ask_price, half: float, timestamp: int) -> None:
        tick = self.cfg.tick_size
        q = self.inventory - self.cfg.target_inventory
        size = self.quoter.order_size(q)

        if bid_price is not None:
            bid_price = math.floor(bid_price / tick) * tick
            self._active_bid_id = self.exchange.place_limit_order("buy", bid_price, size, timestamp)
            self._active_bid_px = bid_price
            self._quotes_placed += 1
            self._quote_log.append({"timestamp": timestamp, "side": "bid", "price": bid_price, "size": size})
            logger.info(f"ORDER buy {size:.6f} BTC @ {bid_price:.2f}")

        if ask_price is not None:
            ask_price = math.ceil(ask_price / tick) * tick
            self._active_ask_id = self.exchange.place_limit_order("sell", ask_price, size, timestamp)
            self._active_ask_px = ask_price
            self._quotes_placed += 1
            self._quote_log.append({"timestamp": timestamp, "side": "ask", "price": ask_price, "size": size})
            logger.info(f"ORDER sell {size:.6f} BTC @ {ask_price:.2f}")

        self._active_half_spread = half

    def on_tick(self, row: dict) -> None:
        ts  = row["timestamp"]
        mid = (row["bid_0_price"] + row["ask_0_price"]) / 2.0

        self.kappa_calib.update_mid(row["bid_0_price"], row["ask_0_price"], ts)
        sigma, vol_ratio = self.vol_est.update(mid, ts)

        if not self.vol_est.ready:
            if self._tick_count % 10 == 0:
                logger.info(f"warming up volatility... {self.vol_est.samples}/{self.vol_est.warmup}")
            self._tick_count += 1
            return

        # Vol spike: cancel quotes immediately and force κ recalibration
        if vol_ratio > self.cfg.vol_spike_threshold:
            logger.info(f"[vol spike] ratio={vol_ratio:.2f} — cancelling quotes")
            self._cancel_stale_quotes()
            self.kappa_calib.flush_sample(ts)

        # Periodic κ recalibration
        self._calib_tick += 1
        if self._calib_tick % self.cfg.kappa_recalib_ticks == 0:
            self.kappa_calib.flush_sample(ts)
            if self.kappa_calib.ready:
                self.cfg.kappa = self.kappa_calib.kappa
                logger.info(f"[calib] κ={self.cfg.kappa:.4f}  α={self.kappa_calib.alpha:.6f} BTC/s")

        fills = self.exchange.check_fills(row)
        self._process_fills(fills)
        if fills:
            self._cancel_stale_quotes()

        self._tick_count += 1
        if self._tick_count % self.cfg.quote_refresh_ticks == 0:
            q = self.inventory - self.cfg.target_inventory
            bid_px, ask_px, half = self.quoter.quotes(mid, q, sigma, vol_ratio)

            if self._quotes_stale(bid_px, ask_px, half):
                self._cancel_stale_quotes()
                self._place_quotes(bid_px, ask_px, half, ts)

            if self._tick_count % 200 == 0:
                self._log_status(mid, bid_px, ask_px, half, sigma, q, vol_ratio)

        self._log.append({
            "timestamp": ts,
            "mid":       mid,
            "inventory": self.inventory,
            "sigma":     sigma,
            "vol_ratio": vol_ratio,
            "kappa":     self.cfg.kappa,
        })

    def _log_status(self, mid, bid_px, ask_px, half, sigma, q, vol_ratio):
        tick = self.cfg.tick_size
        pnl  = self.exchange.realized_pnl(self.inventory, mid)
        bid_d = round(math.floor(bid_px / tick) * tick, 2) if bid_px else "N/A"
        ask_d = round(math.ceil(ask_px / tick) * tick, 2) if ask_px else "N/A"
        fill_rate = (self._fills_bid + self._fills_ask) / max(self._quotes_placed, 1) * 100
        logger.info(
            f"\n{'-'*38}\n"
            f"  mid         = ${mid:.2f}\n"
            f"  bid / ask   = ${bid_d} / ${ask_d}\n"
            f"  half spread = ${half:.2f}\n"
            f"  sigma       = ${sigma:.4f}\n"
            f"  vol ratio   = {vol_ratio:.3f}\n"
            f"  tau         = {self.quoter.time_factor(vol_ratio):.3f}\n"
            f"  gamma       = {self.quoter.effective_gamma(mid, q, sigma):.4f}\n"
            f"  kappa       = {self.cfg.kappa:.3f}\n"
            f"  size        = {self.quoter.order_size(q):.6f} BTC\n"
            f"  inventory   = {self.inventory:+.4f} BTC\n"
            f"  PnL         = ${pnl:+.4f}\n"
            f"  fills       = {self._fills_bid}b / {self._fills_ask}a\n"
            f"  fill rate   = {fill_rate:.1f}%\n"
            f"{'-'*38}\n"
        )

    def run(self, df: pd.DataFrame) -> dict:
        logger.info(f"Running backtest on {len(df)} ticks…")
        for row in df.to_dict("records"):
            self.on_tick(row)
        self.exchange.cancel_all()
        final_mid = (df.iloc[-1]["bid_0_price"] + df.iloc[-1]["ask_0_price"]) / 2.0
        summary = self.exchange.summary()
        summary["realized_pnl"]    = round(self.exchange.realized_pnl(self.inventory, final_mid), 6)
        summary["final_inventory"] = round(self.inventory, 6)
        return summary

    def log_df(self)       -> pd.DataFrame: return pd.DataFrame(self._log)
    def fill_log_df(self)  -> pd.DataFrame: return pd.DataFrame(self._fill_log)
    def quote_log_df(self) -> pd.DataFrame: return pd.DataFrame(self._quote_log)


async def run_live(symbol: str = "btcusdt", cfg: ASConfig = None):
    from orderbook import OrderBookManager
    import websockets, json

    cfg = cfg or ASConfig()
    mm  = MarketMaker(cfg)
    mgr = OrderBookManager(symbol)
    logger.info(f"Running A-S market maker  γ={cfg.gamma}  κ={cfg.kappa}")

    async def depth_loop():
        ws_url = f"wss://fstream.binance.com/ws/{symbol}@depth@100ms"
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
            for _ in range(10):
                mgr.buffer.append(json.loads(await ws.recv()))
            import asyncio as _asyncio
            await _asyncio.get_event_loop().run_in_executor(None, mgr.fetch_snapshot)
            mgr.apply_buffered_updates(mgr.buffer)
            logger.info(f"Order book initialised for {symbol.upper()}")
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("u", 0) <= mgr.last_update_id:
                    continue
                mgr.apply_update(msg)
                row = mgr.flatten()
                if row:
                    mm.on_tick(row)

    async def trade_loop():
        ws_url = f"wss://fstream.binance.com/market/ws/{symbol}@aggTrade"
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
            async for raw in ws:
                msg = json.loads(raw)
                mm.kappa_calib.on_trade(float(msg["p"]), float(msg["q"]), int(msg["T"]))

    await asyncio.gather(depth_loop(), trade_loop())


def run_backtest(parquet_path: str, cfg: ASConfig = None, n_rows: int = 10_000) -> dict:
    cfg = cfg or ASConfig()
 
    if not os.path.exists(parquet_path):
        from stream import stream
        logger.info(f"No data at {parquet_path} — collecting {n_rows} rows...")
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        df = asyncio.run(stream("btcusdt", n_rows=n_rows))
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
    cfg  = ASConfig()
    if args.mode == "live":
        asyncio.run(run_live("btcusdt", cfg))
    else:
        run_backtest("data/raw/book_1777480982.parquet", cfg, n_rows=10_000)
 
