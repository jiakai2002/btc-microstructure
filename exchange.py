from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:
    id: int
    side: str
    price: float
    size: float
    placed_at: int
    filled_size: float = 0.0
    status: str = "open"


@dataclass
class Fill:
    order_id: int
    side: str
    price: float
    size: float
    fee: float
    timestamp: int


class Exchange:

    def __init__(self, fee_bps: float = 2.0):
        self.fee_rate = fee_bps / 10_000
        self._orders: dict[int, Order] = {}
        self._next_id = 0
        self.fills: list[Fill] = []
        self.cancelled: list[Order] = []
        self._cash_flow = 0.0

    def place_limit_order(self, side: str, price: float, size: float,
                          timestamp: int) -> int:
        order = Order(id=self._next_id, side=side, price=price,
                      size=size, placed_at=timestamp)
        self._orders[self._next_id] = order
        self._next_id += 1
        return order.id

    def cancel_order(self, order_id: int) -> bool:
        order = self._orders.get(order_id)
        if order and order.status == "open":
            order.status = "cancelled"
            self.cancelled.append(order)
            del self._orders[order_id]
            return True
        return False

    def cancel_all(self):
        for order_id in list(self._orders.keys()):
            self.cancel_order(order_id)

    def check_fills(self, row: dict) -> list[Fill]:
        best_bid_px = row["bid_0_price"]
        best_ask_px = row["ask_0_price"]
        ts          = row["timestamp"]

        new_fills = []
        for order_id in list(self._orders.keys()):
            order = self._orders.get(order_id)
            if order is None:
                continue

            if order.side == "buy" and best_ask_px <= order.price:
                fill = self._execute(order, order.price, ts)
            elif order.side == "sell" and best_bid_px >= order.price:
                fill = self._execute(order, order.price, ts)
            else:
                fill = None

            if fill:
                new_fills.append(fill)

        return new_fills

    def _execute(self, order: Order, fill_price: float,
                 timestamp: int) -> Optional[Fill]:
        remaining = order.size - order.filled_size
        if remaining <= 0:
            return None

        filled_size = remaining
        fee = filled_size * fill_price * self.fee_rate
        fill = Fill(order_id=order.id, side=order.side, price=fill_price,
                    size=filled_size, fee=fee, timestamp=timestamp)

        if order.side == "sell":
            self._cash_flow += filled_size * fill_price - fee
        else:
            self._cash_flow -= filled_size * fill_price + fee

        order.filled_size += filled_size
        self.fills.append(fill)
        order.status = "filled"
        del self._orders[order.id]
        return fill

    @property
    def open_orders(self) -> list[Order]:
        return list(self._orders.values())

    @property
    def total_fees_paid(self) -> float:
        return sum(f.fee for f in self.fills)

    def realized_pnl(self, current_inventory: float, current_mid: float) -> float:
        return self._cash_flow + current_inventory * current_mid

    def summary(self) -> dict:
        return {
            "total_fills":         len(self.fills),
            "total_fees_paid_usd": round(self.total_fees_paid, 6),
            "open_orders":         len(self._orders),
            "cancelled_orders":    len(self.cancelled),
        }
