import asyncio
import json
import os
import pandas as pd
import websockets
from orderbook import OrderBookManager


async def stream(symbol="btcusdt", n_rows=500):
    manager = OrderBookManager(symbol)
    rows = []
    ws_url = f"wss://fstream.binance.com/ws/{symbol}@depth@100ms"

    async with websockets.connect(ws_url) as ws:
        print("Buffering updates...")
        for _ in range(10):
            msg = await ws.recv()
            manager.buffer.append(json.loads(msg))

        print("Fetching snapshot...")
        manager.fetch_snapshot()
        manager.apply_buffered_updates(manager.buffer)

        print("Applying live updates...")
        while len(rows) < n_rows:
            msg = json.loads(await ws.recv())

            if msg.get("u", 0) <= manager.last_update_id:
                continue

            manager.apply_update(msg)
            row = manager.flatten()
            if row:
                rows.append(row)
                if len(rows) % 100 == 0:
                    print(f"  {len(rows)}/{n_rows}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print("Streaming data...")
    df = asyncio.run(stream("ethusdt", n_rows=1000))
    df.to_parquet("data/raw/book.parquet")
    print(f"Saved {len(df)} rows to data/raw/book.parquet")