import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ASConfig:
    # Core A-S parameters
    gamma: float = 0.1          # fallback when dynamic_gamma=False
    kappa: float = 1.5

    # Continuous-market time factor (replaces session_minutes)
    # infinite_horizon=True → τ=1, correct for 24/7 crypto
    # infinite_horizon=False → vol-driven decay via tau_decay
    infinite_horizon: bool = True
    tau_decay: float = 0.5      # sensitivity when infinite_horizon=False

    # Dynamic gamma: recomputed each cycle from spread bounds + inventory
    dynamic_gamma: bool = True
    inventory_risk_aversion: float = 0.5   # IRA ∈ (0, 1]
    gamma_cap: float = 2.0

    # Spread bounds used by dynamic gamma
    min_spread: float = 0.50
    max_spread: float = 20.0

    # Volatility estimator
    vol_horizon_sec: float = 1.0
    vol_cap: float = 0.5
    vol_floor: float = 1e-6
    vol_spike_threshold: float = 2.0   # vol_ratio above this → emergency cancel

    # Inventory
    max_inventory: float = 0.05
    target_inventory: float = 0.0

    # Order sizing — eta_decay=0 keeps constant size
    order_size: float = 0.001
    eta_decay: float = 0.0

    tick_size: float = 0.10
    quote_refresh_ticks: int = 1

    # κ calibration
    kappa_sampling_length: int = 30
    kappa_min_samples: int = 10
    kappa_recalib_ticks: int = 100


class ASQuoter:
    def __init__(self, cfg: ASConfig):
        self.cfg = cfg

    def time_factor(self, vol_ratio: float = 1.0) -> float:
        """τ=1 for infinite horizon; vol-driven decay otherwise."""
        if self.cfg.infinite_horizon:
            return 1.0
        return math.exp(-self.cfg.tau_decay * max(vol_ratio - 1.0, 0.0))

    def effective_gamma(self, mid: float, q: float, sigma: float) -> float:
        """γ recomputed each cycle so it scales with vol and inventory skew."""
        if not self.cfg.dynamic_gamma or abs(q) < 1e-8:
            return self.cfg.gamma
        sigma_log = sigma / max(mid, 1e-8)
        spread_range = self.cfg.max_spread - self.cfg.min_spread
        gamma_max = spread_range / max(2.0 * abs(q) * sigma_log ** 2, 1e-12)
        return min(gamma_max * self.cfg.inventory_risk_aversion, self.cfg.gamma_cap)

    def order_size(self, q: float) -> float:
        """Size decays with |q| when eta_decay > 0 (Fushimi et al. 2018)."""
        if self.cfg.eta_decay <= 0:
            return self.cfg.order_size
        return self.cfg.order_size * math.exp(-self.cfg.eta_decay * abs(q))

    def reservation_price(self, mid: float, q: float, sigma: float,
                          vol_ratio: float = 1.0) -> float:
        gamma = self.effective_gamma(mid, q, sigma)
        sigma_log = sigma / max(mid, 1e-8)
        tau = self.time_factor(vol_ratio)
        return mid - q * gamma * (sigma_log ** 2) * tau

    def optimal_spread(self, mid: float, q: float, sigma: float,
                       vol_ratio: float = 1.0) -> float:
        gamma = self.effective_gamma(mid, q, sigma)
        sigma_log = sigma / max(mid, 1e-8)
        tau = self.time_factor(vol_ratio)
        kappa = self.cfg.kappa
        term1 = gamma * (sigma_log ** 2) * tau
        term2 = (2.0 / gamma) * math.log(1.0 + gamma / kappa)
        return max((term1 + term2) / 2.0, self.cfg.min_spread)

    def quotes(self, mid: float, q: float, sigma: float,
               vol_ratio: float = 1.0) -> tuple[Optional[float], Optional[float], float]:
        """Returns (bid_price, ask_price, half_spread)."""
        r = self.reservation_price(mid, q, sigma, vol_ratio)
        half = self.optimal_spread(mid, q, sigma, vol_ratio)
        bid_price = r - half
        ask_price = r + half
        if q >= self.cfg.max_inventory:
            bid_price = None
        if q <= -self.cfg.max_inventory:
            ask_price = None
        return bid_price, ask_price, half
