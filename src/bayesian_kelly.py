import logging
import json
import os
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Calibration data file location
STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'state')
CALIBRATION_FILE = os.path.join(STATE_DIR, 'kelly_calibration.json')


class BayesianKellyCriterion:
    """
    Implements a Bayesian-adjusted Fractional Kelly Criterion for capital allocation.

    Dynamically scales position size based on calibrated win probability
    and expected utility, applying a volatility buffer (fractional Kelly) to
    mitigate tail-risk and estimation error.

    Assumptions:
    - Win probability is estimated from vector_strength via a conservative
      calibration curve, NOT directly equated to vector_strength.
    - The reward/risk ratio is assumed constant (configurable).
    - Kelly fraction is an upper bound on growth rate; fractional Kelly
      reduces variance at cost of sub-optimal growth.
    - With <50 historical trades, probability estimates are uncalibrated
      and the system logs a warning accordingly.
    """

    # Minimum trades needed for empirical calibration
    MIN_CALIBRATION_TRADES = 50

    def __init__(
        self,
        account_equity: float,
        fractional_kelly: float = 0.5,
        reward_risk_ratio: float = 2.0,
        min_vector_strength: float = 0.51
    ) -> None:
        self.account_equity = account_equity
        self.fractional_kelly = fractional_kelly
        self.reward_risk_ratio = reward_risk_ratio
        self.min_vector_strength = min_vector_strength
        self._calibration_data: List[Dict] = []
        self._is_calibrated = False

        # Load persisted calibration data
        self._load_calibration()

        logger.info(f"Initialized Kelly Engine | Equity: {account_equity} | "
                    f"Multiplier: {fractional_kelly} | Calibration trades: {len(self._calibration_data)}")

    def _load_calibration(self) -> None:
        """Load persisted calibration data from disk."""
        if not os.path.exists(CALIBRATION_FILE):
            return
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                data = json.load(f)
            self._calibration_data = data.get('trades', [])
            if len(self._calibration_data) >= self.MIN_CALIBRATION_TRADES:
                self._is_calibrated = True
                logger.info(f"Loaded {len(self._calibration_data)} calibration trades (calibrated)")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load calibration data: {e}")
            self._calibration_data = []

    def _save_calibration(self) -> None:
        """Persist calibration data to disk."""
        os.makedirs(STATE_DIR, exist_ok=True)
        try:
            with open(CALIBRATION_FILE, 'w') as f:
                json.dump({'trades': self._calibration_data}, f)
        except IOError as e:
            logger.error(f"Failed to save calibration data: {e}")

    def _estimate_win_probability(self, vector_strength: float) -> float:
        """
        Map vector_strength to a conservative win probability estimate.

        Default: linear map from [0.51, 1.0] -> [0.51, 0.65]
        This prevents over-betting on uncalibrated signal strength.
        Once enough trades are collected (50+), empirical calibration
        can replace this default.

        Args:
            vector_strength: Raw signal strength in [0, 1]

        Returns:
            Calibrated win probability estimate
        """
        if self._is_calibrated and len(self._calibration_data) >= self.MIN_CALIBRATION_TRADES:
            return self._empirical_calibration(vector_strength)

        # Conservative linear mapping: [0.51, 1.0] -> [0.51, 0.65]
        # This prevents the common mistake of equating signal strength with win probability
        p_min, p_max = 0.51, 0.65
        s_min, s_max = self.min_vector_strength, 1.0
        t = (vector_strength - s_min) / (s_max - s_min + 1e-10)
        t = max(0.0, min(1.0, t))
        p = p_min + t * (p_max - p_min)

        if not self._is_calibrated:
            logger.debug(f"Using uncalibrated probability estimate: strength={vector_strength:.3f} -> p={p:.3f}")

        return p

    def _empirical_calibration(self, vector_strength: float) -> float:
        """
        Empirical calibration from historical trade outcomes.
        Bins trades by vector_strength and computes observed win rate per bin.

        Falls back to conservative estimate if insufficient data in the bin.
        """
        if not self._calibration_data:
            return self._estimate_win_probability(vector_strength)

        # Simple binned calibration (5 bins)
        bins = np.linspace(self.min_vector_strength, 1.0, 6)
        for i in range(len(bins) - 1):
            if bins[i] <= vector_strength < bins[i + 1] or (i == len(bins) - 2 and vector_strength == bins[i + 1]):
                bin_trades = [t for t in self._calibration_data
                              if bins[i] <= t['vector_strength'] < bins[i + 1]]
                if len(bin_trades) >= 10:
                    win_rate = sum(1 for t in bin_trades if t['pnl'] > 0) / len(bin_trades)
                    return max(0.51, min(0.65, win_rate))

        # Not enough data in this bin, use conservative default
        return self._estimate_win_probability.__wrapped__(self, vector_strength) if hasattr(self._estimate_win_probability, '__wrapped__') else 0.55

    def add_calibration_trade(self, vector_strength: float, pnl: float) -> None:
        """
        Record a completed trade for future calibration.

        Args:
            vector_strength: Signal strength at entry
            pnl: Realized P&L of the trade
        """
        self._calibration_data.append({
            'vector_strength': vector_strength,
            'pnl': pnl
        })
        self._save_calibration()

        if len(self._calibration_data) >= self.MIN_CALIBRATION_TRADES and not self._is_calibrated:
            self._is_calibrated = True
            logger.info(f"Kelly calibration activated with {len(self._calibration_data)} trades")

    def calculate_kelly_fraction(self, vector_strength: float) -> float:
        """
        Derives the optimal growth fraction f* using the Kelly formula:
        f* = (p*b - q) / b

        Where p is the calibrated win probability (not raw vector_strength).
        """
        if vector_strength < self.min_vector_strength:
            return 0.0

        p = self._estimate_win_probability(vector_strength)
        q = 1.0 - p
        b = self.reward_risk_ratio

        raw_kelly = (p * b - q) / b

        if raw_kelly <= 0:
            return 0.0

        # Apply fractional safety buffer and hard concentration cap
        safe_kelly = raw_kelly * self.fractional_kelly
        return max(0.0, min(safe_kelly, 0.25))

    def calculate_position_size(
        self,
        vector_strength: float,
        risk_per_share: float,
        buying_power: float,
        current_price: float,
        max_concentration: float = 0.20
    ) -> int:
        """
        Returns share quantity constrained by Kelly allocation, available
        liquidity, and portfolio concentration limits.

        Args:
            vector_strength: Signal strength [0, 1]
            risk_per_share: Dollar risk per share (entry - stop)
            buying_power: Available buying power in dollars
            current_price: Current share price for liquidity calculation
            max_concentration: Maximum portfolio % in single position
        """
        f_star = self.calculate_kelly_fraction(vector_strength)

        if f_star == 0:
            return 0

        if current_price <= 0:
            return 0

        # Kelly sizing: f* of equity allocated to RISK, not position value
        # If we risk f* of equity, and risk_per_share is our risk, then:
        # shares = (equity * f*) / risk_per_share
        kelly_risk_budget = self.account_equity * f_star
        shares_kelly = kelly_risk_budget / (risk_per_share + 1e-6)

        # Concentration limit: max % of equity in position VALUE
        max_position_value = self.account_equity * max_concentration
        shares_concentration = max_position_value / current_price

        # Liquidity limit: can't spend more than buying power
        shares_liquid = buying_power / current_price

        qty = int(min(shares_kelly, shares_concentration, shares_liquid))

        logger.info(f"Sizing | strength={vector_strength:.2f} f*={f_star:.4f} "
                    f"kelly={shares_kelly:.0f} conc={shares_concentration:.0f} "
                    f"liquid={shares_liquid:.0f} -> qty={qty}")
        return max(0, qty)

    def get_expected_value(
        self,
        vector_strength: float,
        entry: float,
        stop: float,
        target: float
    ) -> Dict:
        """
        Calculates probabilistic trade expectancy (EV) using calibrated probability.
        """
        p_win = self._estimate_win_probability(vector_strength)
        p_loss = 1.0 - p_win

        win_amt = target - entry
        loss_amt = entry - stop

        ev = (p_win * win_amt) - (p_loss * loss_amt)

        return {
            'ev': ev,
            'is_favorable': ev > 0 and vector_strength >= self.min_vector_strength,
            'calibrated_p_win': p_win,
            'raw_strength': vector_strength
        }
