import numpy as np
import pandas as pd

from exchange import SimulatedExchange


class MarketMaker:
    def __init__(
        self,
        base_spread_bps: float = 2.0,
        order_size: float      = 0.001,
        max_inventory: float   = 0.1,
        skew_factor: float     = 0.5,
        vol_window: int        = 50,
        vol_scalar: float      = 5000.0,
        obi_factor: float      = 0.1,
        fee_bps: float         = 2.0,
    ):
        self.base_spread_bps = base_spread_bps
        self.order_size      = order_size
        self.max_inventory   = max_inventory
        self.skew_factor     = skew_factor
        self.vol_window      = vol_window
        self.vol_scalar      = vol_scalar
        self.obi_factor      = obi_factor

        self.exchange        = SimulatedExchange(fee_bps=fee_bps)
        self.inventory       = 0.0
        self._mid_prices: list[float] = []

    def update_volatility(self, mid: float) -> float:
        self._mid_prices.append(mid)
        if len(self._mid_prices) > self.vol_window:
            self._mid_prices.pop(0)
        if len(self._mid_prices) < 2:
            return 0.0
        returns = np.diff(np.log(self._mid_prices))
        return float(np.std(returns))

    def compute_obi(self, row: dict) -> float:
        bid_vol = sum(row[f"bid_{i}_size"] for i in range(5))
        ask_vol = sum(row[f"ask_{i}_size"] for i in range(5))
        total   = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def generate_quotes(self, mid: float, vol: float, obi: float) -> tuple[float, float]:
        # 1. volatility-adjusted spread
        half_spread = mid * (self.base_spread_bps / 10000) / 2
        half_spread += vol * self.vol_scalar

        # 2. inventory skew — lean against position
        # if long inventory: lower bid/ask -> ask more competetive -> more ppl buy from us
        # if short inventory: raise bid/ask -> bids more competetive -> more ppl sell to us
        inventory_skew = (self.inventory / self.max_inventory) * self.skew_factor * half_spread
        obi_skew       = obi * self.obi_factor * half_spread

        # 3. obi signal skew — lean into predicted direction
        bid = mid - half_spread - inventory_skew + obi_skew
        ask = mid + half_spread - inventory_skew + obi_skew

        return bid, ask

    def step(self, row: dict, prev_row: dict | None) -> dict:
        mid = (row["bid_0_price"] + row["ask_0_price"]) / 2

        # 1. update vol and bid
        vol = self.update_volatility(mid)
        obi = self.compute_obi(row)

        # 2. check fills on previous step's orders
        fills = self.exchange.check_fills(prev_row) if prev_row else []

        # 3. update inventory
        for fill in fills:
            if fill.side == "buy":
                self.inventory += fill.size
            else:
                self.inventory -= fill.size

        # 4. cancel stale orders
        self.exchange.cancel_all()

        # 5. generate quotes and place order (skip if inventory limit reached) 
        if abs(self.inventory) < self.max_inventory:
            bid, ask = self.generate_quotes(mid, vol, obi)
            self.exchange.place_limit_order("buy",  bid, self.order_size, row["timestamp"])
            self.exchange.place_limit_order("sell", ask, self.order_size, row["timestamp"])

        # 6. record snapshot of market maker's state
        return {
            "timestamp": row["timestamp"],
            "mid":       mid,
            "vol":       vol,
            "obi":       obi,
            "inventory": self.inventory,
            "pnl":       self.exchange.realized_pnl(self.inventory, mid),
        }

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        snapshots = []
        prev_row  = None

        for _, row in df.iterrows():
            row      = row.to_dict()
            snapshot = self.step(row, prev_row)
            snapshots.append(snapshot)
            prev_row = row

        return pd.DataFrame(snapshots)