import asyncio
import json
import os
import time

import pandas as pd
import websockets

from orderbook import OrderBookManager


async def _stream_session(symbol: str, n_rows: int, rows: list) -> bool:
    """
    Single WebSocket session. Returns True if target reached, False if disconnected.
    Appends rows in-place so progress is preserved across reconnects.
    """
    manager = OrderBookManager(symbol)
    ws_url  = f"wss://fstream.binance.com/ws/{symbol}@depth@100ms"

    try:
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
            print("Buffering updates...")
            for _ in range(10):
                msg = await ws.recv()
                manager.buffer.append(json.loads(msg))

            print("Fetching snapshot...")
            manager.fetch_snapshot()
            manager.apply_buffered_updates(manager.buffer)

            print(f"Collecting rows ({len(rows)}/{n_rows} so far)...")
            while len(rows) < n_rows:
                msg = json.loads(await ws.recv())

                if msg.get("u", 0) <= manager.last_update_id:
                    continue

                manager.apply_update(msg)
                row = manager.flatten()
                if row:
                    rows.append(row)
                    if len(rows) % 1000 == 0:
                        print(f"  {len(rows)}/{n_rows}")

        return True

    except (websockets.ConnectionClosed, asyncio.TimeoutError, Exception) as e:
        print(f"Session ended: {e}")
        return False


async def stream(symbol: str = "btcusdt", n_rows: int = 500, max_retries: int = 10) -> pd.DataFrame:
    rows    = []
    retries = 0

    while len(rows) < n_rows and retries < max_retries:
        done = await _stream_session(symbol, n_rows, rows)
        if done:
            break
        retries += 1
        wait = min(5 * retries, 60)  # backoff: 5s, 10s, 15s ... capped at 60s
        print(f"Reconnecting in {wait}s (attempt {retries}/{max_retries})...")
        await asyncio.sleep(wait)

    if len(rows) < n_rows:
        print(f"Warning: collected {len(rows)}/{n_rows} rows after {retries} retries")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    SYMBOL = "btcusdt"
    N_ROWS = 1_000_000    # ~3 hours at 100ms

    os.makedirs("data/raw", exist_ok=True)
    out_path = f"data/raw/book_{int(time.time())}.parquet"

    print(f"Streaming {SYMBOL.upper()} — target {N_ROWS} rows...")
    df = asyncio.run(stream(SYMBOL, n_rows=N_ROWS))
    df.to_parquet(out_path)
    print(f"Saved {len(df)} rows to {out_path}")