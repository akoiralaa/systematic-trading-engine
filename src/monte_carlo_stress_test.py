import numpy as np
from scipy import stats
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MonteCarloStressTest:
    """
    Stochastic risk engine for strategy validation and tail-risk quantification.

    Utilizes bootstrap resampling to generate multi-path equity distributions,
    quantifying the 'Risk of Ruin' and 'Conditional Value at Risk' (CVaR) under
    non-Gaussian return assumptions.

    Assumptions:
    - Default bootstrap assumes IID trade returns (each trade is independent).
      This is arguably valid for trade-level returns where positions don't
      overlap, but NOT valid for bar-level returns with serial correlation.
    - For bar-level returns, use block_size > 1 to preserve temporal structure
      via the stationary block bootstrap (Politis & Romano, 1994).
    - Equity paths are computed as cumulative products, assuming multiplicative
      compounding.
    - Shock injection (stress_test_shocks) models independent exogenous events,
      not endogenous market dynamics.
    """

    def __init__(self, initial_equity: float = 100000, simulations: int = 10000) -> None:
        self.initial_equity = initial_equity
        self.simulations = simulations
        logger.info(f"RiskEngine: Equity={initial_equity}, Iterations={simulations}")

    def _block_bootstrap(self, data: np.ndarray, n_samples: int, block_size: int = 1) -> np.ndarray:
        """
        Generate bootstrap samples, optionally using block bootstrap.

        Args:
            data: Original return series
            n_samples: Number of elements per sample
            block_size: Size of contiguous blocks to preserve temporal structure.
                        block_size=1 is equivalent to standard IID bootstrap.

        Returns:
            Array of shape (self.simulations, n_samples) with resampled returns
        """
        if block_size <= 1:
            # Standard IID bootstrap
            indices = np.random.randint(0, len(data), size=(self.simulations, n_samples))
            return data[indices]

        # Block bootstrap (Politis & Romano, 1994)
        n_blocks = (n_samples + block_size - 1) // block_size
        max_start = len(data) - block_size
        if max_start < 1:
            max_start = 1

        result = np.zeros((self.simulations, n_samples))
        for sim in range(self.simulations):
            blocks = []
            for _ in range(n_blocks):
                start = np.random.randint(0, max_start + 1)
                blocks.append(data[start:start + block_size])
            path = np.concatenate(blocks)[:n_samples]
            result[sim] = path

        return result

    def run_probability_cone(self, trade_returns: np.ndarray,
                             confidence_levels: Optional[List[int]] = None,
                             block_size: int = 1) -> Dict:
        """
        Generates a fan chart of potential equity trajectories via bootstrap sampling.

        Args:
            trade_returns: Array of historical returns
            confidence_levels: Percentile levels for the cone (default [5,25,50,75,95])
            block_size: Block size for bootstrap (1=IID, >1=block bootstrap)

        Returns:
            Dict with equity paths, percentiles, and summary statistics
        """
        if confidence_levels is None:
            confidence_levels = [5, 25, 50, 75, 95]

        path_returns = self._block_bootstrap(trade_returns, len(trade_returns), block_size)
        equity_array = np.cumprod(1 + path_returns, axis=1) * self.initial_equity

        percentiles = {f'p{conf}': np.percentile(equity_array, conf, axis=0) for conf in confidence_levels}
        final_dist = equity_array[:, -1]

        return {
            'paths': equity_array,
            'final_equity_dist': final_dist,
            'percentiles': percentiles,
            'p5_worst_case': np.percentile(final_dist, 5),
            'p50_median': np.percentile(final_dist, 50),
            'p95_best_case': np.percentile(final_dist, 95),
            'terminal_std': np.std(final_dist)
        }

    def calculate_risk_of_ruin(self, trade_returns: np.ndarray,
                                ruin_threshold: float = 0.20,
                                block_size: int = 1) -> Dict:
        """
        Quantifies the probability of crossing a terminal drawdown threshold.

        P(E_final < E_0 * (1 - threshold))
        """
        path_returns = self._block_bootstrap(trade_returns, len(trade_returns), block_size)
        final_equities = np.prod(1 + path_returns, axis=1) * self.initial_equity

        ruin_level = self.initial_equity * (1 - ruin_threshold)
        ruin_count = np.sum(final_equities < ruin_level)
        ror = ruin_count / self.simulations

        logger.warning(f"TailRisk: RiskOfRuin={ror*100:.2f}% | Threshold={ruin_threshold}")

        return {
            'risk_of_ruin_pct': ror * 100,
            'threshold_value': ruin_level,
            'failure_count': int(ruin_count)
        }

    def stress_test_shocks(self, trade_returns: np.ndarray, shock_mag: float = -0.10,
                           shock_prob: float = 0.10) -> Dict:
        """
        Performs kurtosis injection to simulate 'Black Swan' events.

        Artificially introduces low-probability, high-severity negative returns
        into the return distribution to test system robustness against tail events.
        """
        final_equities = []
        max_dds = []

        for _ in range(self.simulations):
            path = np.random.choice(trade_returns, size=len(trade_returns), replace=True)

            # Stochastic shock injection
            if np.random.random() < shock_prob:
                path[np.random.randint(0, len(path))] = shock_mag

            equity_curve = np.cumprod(1 + path) * self.initial_equity
            final_equities.append(equity_curve[-1])
            max_dds.append(np.min(equity_curve / self.initial_equity) - 1)

        return {
            'shock_survival_rate': np.sum(np.array(final_equities) > (self.initial_equity * 0.8)) / self.simulations,
            'median_drawdown': np.median(max_dds),
            'tail_drawdown_p5': np.percentile(max_dds, 5)
        }

    def get_tail_risk_metrics(self, trade_returns: np.ndarray, alpha: float = 0.95,
                              block_size: int = 1) -> Dict:
        """
        Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR).

        CVaR (Expected Shortfall) provides the average loss in the (1-alpha)
        worst cases, offering a superior risk measure for non-normal distributions.
        """
        path_returns = self._block_bootstrap(trade_returns, len(trade_returns), block_size)
        final_dist = np.prod(1 + path_returns, axis=1) * self.initial_equity

        # VaR is the (1-alpha) percentile of the final equity distribution
        var_threshold = np.percentile(final_dist, (1 - alpha) * 100)

        # CVaR is the expectation of the distribution below the VaR threshold
        tail = final_dist[final_dist <= var_threshold]
        cvar = tail.mean() if len(tail) > 0 else var_threshold

        return {
            'var_alpha': var_threshold,
            'cvar_expected_shortfall': cvar,
            'max_expected_loss_pct': ((self.initial_equity - cvar) / self.initial_equity) * 100
        }
