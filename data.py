import asyncio
import websockets
import json
import time
import requests
import pandas as pd
import os
from sortedcontainers import SortedDict


class OrderBookManager:
    def __init__(self, symbol="btcusdt"):
        self.symbol = symbol
        self.bids = SortedDict(lambda x: -x)  # descending — best bid first
        self.asks = SortedDict()               # ascending — best ask first
        self.last_update_id = None
        self.buffer = []
        self.ready = False

    def fetch_snapshot(self):
        # get snapshot n load into order book
        url = f"https://fapi.binance.com/fapi/v1/depth?symbol={self.symbol.upper()}&limit=50"
        r = requests.get(url).json()
        self.last_update_id = r["lastUpdateId"]
        self.bids.clear()
        self.asks.clear()
        for price, size in r["bids"]:
            self.bids[float(price)] = float(size)
        for price, size in r["asks"]:
            self.asks[float(price)] = float(size)
        self.ready = True
        print(f"Snapshot loaded. Last update ID: {self.last_update_id}")

    def apply_update(self, data):
        # process delta update
        for price, size in data.get("b", []):
            price, size = float(price), float(size)
            if size == 0:
                self.bids.pop(price, None)  # level dissapeared so remove it
            else:
                self.bids[price] = size     # level updated so update size

        for price, size in data.get("a", []):
            price, size = float(price), float(size)
            if size == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = size
    
    def apply_buffered_updates(self, buffer):
        for event in buffer:
            # skip old updates
            if event.get("u", 0) <= self.last_update_id:
                continue

            # gap detected - need to resync
            if event.get("U", 0) > self.last_update_id + 1:
                raise Exception("Order book out of sync, need resync")

            self.apply_update(event)
            self.last_update_id = event["u"]

    def top_n(self, n=5):
        bids = list(self.bids.items())[:n]
        asks = list(self.asks.items())[:n]
        return bids, asks

    def flatten(self, n=5):
        bids, asks = self.top_n(n)
        if len(bids) < n or len(asks) < n:
            return None
        row = {
            "timestamp": int(time.time() * 1000),
            "spread": asks[0][0] - bids[0][0],
        }
        for i in range(n):
            row[f"bid_{i}_price"] = bids[i][0]
            row[f"bid_{i}_size"] = bids[i][1]
            row[f"ask_{i}_price"] = asks[i][0]
            row[f"ask_{i}_size"] = asks[i][1]
        return row

async def stream(symbol="btcusdt", n_rows=500):
    manager = OrderBookManager(symbol)
    rows = []
    ws_url = f"wss://fstream.binance.com/ws/{symbol}@depth@100ms"

    async with websockets.connect(ws_url) as ws:
        # buffer incoming updates before snapshot
        print("Buffering updates...")
        for _ in range(10):
            msg = await ws.recv()
            manager.buffer.append(json.loads(msg))

        # load snapshot into book
        print("Fetch snapshot...")
        manager.fetch_snapshot()

        # apply buffered updates to snapshot
        manager.apply_buffered_updates(manager.buffer)

        print("Applying live updates...")
        while len(rows) < n_rows:
            msg = json.loads(await ws.recv())

            # discard old updates
            if msg.get("u", 0) <= manager.last_update_id:
                continue

            manager.apply_update(msg)
            # record at event-driven intervals so irregular time-series
            row = manager.flatten()
            if row:
                rows.append(row)
                if len(rows) % 100 == 0:
                    print(f"  {len(rows)}/{n_rows}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    print("Streaming BTCUSDT...")
    df = asyncio.run(stream("btcusdt", n_rows=5000))
    df.to_parquet("data/raw/book.parquet")
    print(f"Saved {len(df)} rows to data/raw/book.parquet")
