import logging
from typing import Dict

logger = logging.getLogger(__name__)

class MarketFrictionModel:
    """
    Implements a non-linear market impact and transaction cost model.
    
    Estimates the implementation shortfall by modeling the 'walk-the-book' effect 
    using a power-law function of order size relative to Average Daily Volume (ADV).
    Incorporates fixed bid-ask spread costs and dynamic liquidity-driven slippage.
    """
    
    def __init__(self, market_impact_coeff: float = 0.1, bid_ask_spread_bps: float = 2.0) -> None:
        """
        Initializes the friction model with calibrated liquidity parameters.
        
        Args:
            market_impact_coeff: Scaling factor for non-linear price impact.
            bid_ask_spread_bps: Baseline bid-ask spread in basis points.
        """
        self.market_impact_coeff = market_impact_coeff
        self.bid_ask_spread_bps = bid_ask_spread_bps
        logger.info(f"FrictionEngine: ImpactCoeff={market_impact_coeff}, BaseSpread={bid_ask_spread_bps}bps")
    
    def calculate_dynamic_slippage(
        self, qty: int, avg_volume: float, price: float
    ) -> Dict[str, float]:
        """
        Calculates expected slippage via a non-linear volume-participation model.
        
        Uses the power-law impact formula:
        $$I = \alpha \cdot (\frac{V_{order}}{V_{ADV}})^{1.5}$$
        Where $\alpha$ is the market_impact_coeff.
        """
        if avg_volume <= 0 or qty <= 0:
            return {
                'volume_ratio': 0.0,
                'impact_bps': self.bid_ask_spread_bps / 2,
                'execution_price': price
            }
        
        # Calculate participation rate (V_order / V_ADV)
        participation_rate = qty / avg_volume
        
        # Non-linear impact calculation (Square-root/Power-law variant)
        # Models disproportionate impact as order size approaches liquidity ceilings
        impact_bps = ((participation_rate * 100) ** 1.5) * self.market_impact_coeff
        
        # Derive implementation price
        execution_price = price * (1 + impact_bps / 10000)
        
        return {
            'volume_ratio': participation_rate * 100,
            'impact_bps': impact_bps,
            'execution_price': execution_price
        }
    
    def calculate_total_friction(
        self, qty: int, avg_volume: float, price: float, side: str = 'buy'
    ) -> Dict[str, float]:
        """
        Aggregates total transaction costs (Slippage + Half-Spread).
        
        Adjusts the arrival price to the execution price based on order direction 
        to account for the crossing of the bid-ask spread.
        """
        slippage = self.calculate_dynamic_slippage(qty, avg_volume, price)
        
        # Model half-spread crossing cost
        spread_cost_bps = self.bid_ask_spread_bps / 2
        total_friction_bps = slippage['impact_bps'] + spread_cost_bps
        
        # Adjust execution price based on side (Buyer pays premium, Seller accepts discount)
        direction = 1 if side.lower() == 'buy' else -1
        execution_price = price * (1 + (direction * total_friction_bps / 10000))
        
        logger.debug(f"TradeFriction | Side: {side} | TotalBps: {total_friction_bps:.2f} | P_exec: {execution_price:.4f}")
        
        return {
            'bid_ask_bps': spread_cost_bps,
            'impact_bps': slippage['impact_bps'],
            'total_friction_bps': total_friction_bps,
            'execution_price': execution_price
        }
    
    def get_liquidity_constrained_size(
        self, avg_volume: float, max_participation_rate: float = 0.05
    ) -> int:
        """
        Returns the maximum allowable position size based on ADV constraints.

        Adheres to institutional '1-day volume' participation limits to ensure
        order completion within a single session without excessive price distortion.
        """
        max_qty = int(avg_volume * max_participation_rate)
        logger.info(f"LiquidityGuard | MaxSize: {max_qty} (Cap: {max_participation_rate*100}%)")
        return max_qty

    def calculate_exit_price(
        self, qty: int, avg_volume: float, price: float, exit_type: str = 'normal'
    ) -> float:
        """
        Calculate realistic exit price including slippage.

        Args:
            qty: Number of shares to sell
            avg_volume: Average daily volume
            price: Current market price
            exit_type: 'normal', 'stop_loss', or 'urgent'
                       stop_loss/urgent exits have 2x slippage due to adverse conditions

        Returns:
            Expected execution price after friction
        """
        friction = self.calculate_total_friction(qty, avg_volume, price, side='sell')

        # Stop losses and urgent exits typically experience worse fills
        slippage_multiplier = 2.0 if exit_type in ('stop_loss', 'urgent') else 1.0

        total_friction_bps = friction['total_friction_bps'] * slippage_multiplier

        # Seller receives less (negative direction)
        exit_price = price * (1 - total_friction_bps / 10000)

        logger.debug(f"ExitFriction | Type: {exit_type} | Friction: {total_friction_bps:.2f}bps | "
                     f"Price: {price:.2f} -> {exit_price:.2f}")

        return exit_price