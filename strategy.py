import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ASConfig:
    # Core A-S parameters
    gamma: float = 0.1          # used as cap when dynamic_gamma is on
    kappa: float = 1.5

    # Fix 4: dynamic gamma controls
    dynamic_gamma: bool = True
    inventory_risk_aversion: float = 0.5   # IRA ∈ (0, 1]
    gamma_cap: float = 2.0                 # hard upper bound on computed γ

    # Spread bounds (used by dynamic gamma)
    min_spread: float = 0.50
    max_spread: float = 20.0

    # Session
    session_minutes: float = 60.0

    # Volatility estimator
    vol_horizon_sec: float = 1.0
    vol_cap: float = 0.5
    vol_floor: float = 1e-6
    vol_spike_threshold: float = 2.0       # Fix 7: ratio that triggers emergency cancel

    # Inventory
    max_inventory: float = 0.05
    target_inventory: float = 0.0

    # Fix 6: order size shape factor (η)
    order_size: float = 0.001              # base size at q=0
    eta_decay: float = 0.0                 # set > 0 to enable size decay; 0 = off
                                           # eta = 1 / (total_inventory / IRA)
                                           # e.g. with total_inventory=0.1, IRA=0.5 → eta=5

    tick_size: float = 0.10
    quote_refresh_ticks: int = 1

    # κ calibration
    kappa_sampling_length: int = 30
    kappa_min_samples: int = 10
    kappa_recalib_ticks: int = 100


class ASQuoter:
    def __init__(self, cfg: ASConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Time factor
    # ------------------------------------------------------------------

    def time_factor(self, t: float) -> float:
        return max(0.0, min(1.0, 1.0 - t))

    # ------------------------------------------------------------------
    # Fix 4: dynamic gamma
    # ------------------------------------------------------------------

    def effective_gamma(self, mid: float, q: float, sigma: float) -> float:
        """
        Recomputes γ each cycle so it scales with volatility and inventory.
        Falls back to cfg.gamma when dynamic_gamma is off or q ≈ 0.
        """
        if not self.cfg.dynamic_gamma or abs(q) < 1e-8:
            return self.cfg.gamma

        sigma_log = sigma / max(mid, 1e-8)
        abs_q = abs(q)
        spread_range = self.cfg.max_spread - self.cfg.min_spread

        gamma_max = spread_range / max(2.0 * abs_q * sigma_log ** 2, 1e-12)
        gamma = gamma_max * self.cfg.inventory_risk_aversion
        return min(gamma, self.cfg.gamma_cap)

    # ------------------------------------------------------------------
    # Fix 6: order size decay via η
    # ------------------------------------------------------------------

    def order_size(self, q: float) -> float:
        """
        Scales order size down as |q| grows away from target.
        eta_decay = 0 → constant size (original behaviour).
        """
        if self.cfg.eta_decay <= 0:
            return self.cfg.order_size
        return self.cfg.order_size * math.exp(-self.cfg.eta_decay * abs(q))

    # ------------------------------------------------------------------
    # Core A-S formulas
    # ------------------------------------------------------------------

    def reservation_price(self, mid: float, q: float, sigma: float, t: float) -> float:
        gamma = self.effective_gamma(mid, q, sigma)
        sigma_log = sigma / max(mid, 1e-8)
        tau = self.time_factor(t)
        return mid - q * gamma * (sigma_log ** 2) * tau

    def optimal_spread(self, mid: float, q: float, sigma: float, t: float) -> float:
        gamma = self.effective_gamma(mid, q, sigma)
        sigma_log = sigma / max(mid, 1e-8)
        tau = self.time_factor(t)
        kappa = self.cfg.kappa
        term1 = gamma * (sigma_log ** 2) * tau
        term2 = (2.0 / gamma) * math.log(1.0 + gamma / kappa)
        return max((term1 + term2) / 2.0, self.cfg.min_spread)

    def quotes(
        self, mid: float, q: float, sigma: float, t: float
    ) -> tuple[Optional[float], Optional[float], float]:
        """Returns (bid_price, ask_price, half_spread)."""
        r = self.reservation_price(mid, q, sigma, t)
        half = self.optimal_spread(mid, q, sigma, t)
        bid_price = r - half
        ask_price = r + half
        if q >= self.cfg.max_inventory:
            bid_price = None
        if q <= -self.cfg.max_inventory:
            ask_price = None
        return bid_price, ask_price, half
