import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Implements a volatility-adaptive signal filter and market state classifier.
    
    Utilizes ATR-weighted 'Dead Bands' to differentiate between stochastic noise 
    and structural momentum. Employs OLS regression for objective regime 
    classification and statistical significance testing.
    """
    
    def __init__(self, atr_multiplier: float = 2.0, min_vector_strength: float = 0.51) -> None:
        """
        Initializes detector with volatility-scaling and confidence parameters.
        
        Args:
            atr_multiplier: Width of the volatility-adjusted dead band.
            min_vector_strength: Lower bound for signal probability threshold.
        """
        self.atr_multiplier = atr_multiplier
        self.min_vector_strength = min_vector_strength
        logger.info(f"RegimeEngine: ATR_Mult={atr_multiplier}, StrengthThreshold={min_vector_strength}")
    
    def calculate_adaptive_zones(self, prices: np.ndarray, atr: np.ndarray, 
                                vector: np.ndarray, strengths: np.ndarray) -> Dict:
        """
        Derives dynamic dead bands around the primary vector line.
        
        Signals are categorized as valid only upon breaching the +/-(k * ATR)
        threshold, effectively filtering out sub-volatility mean-reversion.
        """
        band_offset = self.atr_multiplier * atr
        
        upper_bound = vector + band_offset
        lower_bound = vector - band_offset
        
        # Binary state mapping for breakout conditions
        valid_mask = ((prices > upper_bound) | (prices < lower_bound)) & (strengths >= self.min_vector_strength)
        
        return {
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'is_valid_signal': valid_mask,
            'filtered_strength': strengths * valid_mask.astype(float)
        }
    
    def detect_regime(self, prices: np.ndarray, lookback: int = 30) -> Dict:
        """
        Classifies the current state of the price process via OLS regression
        on log returns (not raw prices, which are non-stationary).

        State space:
        - TRENDING: Cumulative returns show significant directional bias
        - VOLATILE: Elevated standard deviation of log returns
        - SIDEWAYS: Low-momentum consolidation within stochastic noise
        """
        window = prices[-lookback:]
        log_prices = np.log(window)
        returns = np.diff(log_prices)
        volatility = np.std(returns)

        # OLS on cumulative log returns (stationary transformation)
        cum_returns = np.cumsum(returns)
        x = np.arange(len(cum_returns))
        if len(cum_returns) < 2:
            return {
                'state': 'SIDEWAYS',
                'volatility': volatility,
                'p_value': 1.0,
                'is_significant': False,
                'r_squared': 0.0,
                'trend_strength': 0.0,
                'confidence': 0.0
            }

        slope, _, r_value, p_val, _ = stats.linregress(x, cum_returns)
        r_squared = r_value ** 2

        # Drift: average return per bar
        drift = abs(np.mean(returns))

        # Signal-to-noise ratio
        eps = 1e-10
        trend_strength = abs(slope) / (volatility + eps)

        # Adaptive volatility threshold based on timeframe
        # 0.02 for daily, ~0.005 for 5-min bars
        vol_threshold = 0.02 if volatility > 0.01 else 0.005

        # State classification
        if p_val < 0.05 and abs(slope) > volatility * 0.5:
            state = 'TRENDING'
        elif volatility > vol_threshold:
            state = 'VOLATILE'
        else:
            state = 'SIDEWAYS'

        logger.info(f"MarketState: {state} | Drift: {drift:.5f} | Vol: {volatility:.4f} | "
                    f"R2: {r_squared:.4f} | P: {p_val:.4f}")

        return {
            'state': state,
            'volatility': volatility,
            'p_value': p_val,
            'is_significant': p_val < 0.05,
            'r_squared': r_squared,
            'trend_strength': trend_strength,
            'confidence': r_squared if state == 'TRENDING' else max(0, 1.0 - volatility * 10)
        }

    

    def validate_execution_signal(self, price: float, vector: float, atr: float,
                                 strength: float, state: str, side: str = 'long') -> Dict:
        """
        Performs multi-factor confirmation of directional alpha.

        Validates the confluence of volatility-weighted clearance,
        probabilistic confidence, and favorable macro-state (State != SIDEWAYS).

        For LONG signals: price must be ABOVE the upper band (bullish breakout)
        For SHORT signals: price must be BELOW the lower band (bearish breakout)
        """
        band_width = self.atr_multiplier * atr
        upper_band = vector + band_width
        lower_band = vector - band_width

        if side == 'long':
            # Long signal: price must break above the upper band
            clears_dead_band = price > upper_band
        else:
            # Short signal: price must break below the lower band
            clears_dead_band = price < lower_band

        is_confirmed = (clears_dead_band and
                        strength >= self.min_vector_strength and
                        state != 'SIDEWAYS')

        return {
            'is_confirmed': is_confirmed,
            'dead_band_breach': clears_dead_band,
            'side': side,
            'state': state,
            'quality_score': float(is_confirmed) * strength
        }

    def get_volatility_adjusted_stop(self, entry: float, vector: float,
                                    atr: float, side: str = 'long') -> Dict:
        """
        Calculates dynamic protective stops based on local volatility.

        Adjusts the risk-distance to ensure stops reside outside the
        stochastic noise zone defined by ATR and the structural vector line.
        """
        offset = self.atr_multiplier * atr

        if side == 'long':
            # Stop must be below vector and outside ATR noise
            stop = min(entry - offset, vector - atr)
            # Floor: stop can't be below 5% of entry price (prevents absurd stops)
            min_stop = entry * 0.95
            stop = max(stop, min_stop)
            # Also can't be negative
            stop = max(stop, 0.01)
        else:
            # Stop must be above vector and outside ATR noise
            stop = max(entry + offset, vector + atr)
            # Ceiling: stop can't be above 5% of entry price
            max_stop = entry * 1.05
            stop = min(stop, max_stop)

        risk_per_share = abs(entry - stop)
        # Minimum risk of 0.1% of entry to avoid division issues
        risk_per_share = max(risk_per_share, entry * 0.001)

        return {
            'stop_price': stop,
            'risk_per_share': risk_per_share,
            'volatility_buffer': offset
        }
    def calculate_dynamic_stop(self, price: float, vector: float, atr: float, side: str = 'long') -> Dict:
        """Alias for get_volatility_adjusted_stop to maintain API compatibility."""
        return self.get_volatility_adjusted_stop(price, vector, atr, side)
