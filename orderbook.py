import requests
from sortedcontainers import SortedDict


class OrderBookManager:
    def __init__(self, symbol="btcusdt"):
        self.symbol = symbol
        self.bids = SortedDict(lambda x: -x)
        self.asks = SortedDict()
        self.last_update_id = None
        self.buffer = []
        self.ready = False

    def fetch_snapshot(self):
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
        for price, size in data.get("b", []):
            price, size = float(price), float(size)
            if size == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = size

        for price, size in data.get("a", []):
            price, size = float(price), float(size)
            if size == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = size

    def apply_buffered_updates(self, buffer):
        for event in buffer:
            if event.get("u", 0) <= self.last_update_id:
                continue
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
        import time
        row = {
            "timestamp": int(time.time() * 1000),
            "spread": asks[0][0] - bids[0][0],
        }
        for i in range(n):
            row[f"bid_{i}_price"] = bids[i][0]
            row[f"bid_{i}_size"]  = bids[i][1]
            row[f"ask_{i}_price"] = asks[i][0]
            row[f"ask_{i}_size"]  = asks[i][1]
        return row