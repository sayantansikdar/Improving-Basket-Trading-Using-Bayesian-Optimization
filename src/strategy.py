"""
Trading strategy implementation for basket trading.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from src.utils import z_score


class TradingStrategy:
    """
    Mean-reversion trading strategy for cointegrated baskets.
    """
    
    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        transaction_cost: float = 0.001,
        lookback_period: int = 60
    ):
        """
        Initialize trading strategy.
        
        Parameters:
        -----------
        entry_threshold : float
            Entry threshold in standard deviations
        exit_threshold : float
            Exit threshold in standard deviations (mean reversion)
        transaction_cost : float
            Transaction cost per trade (as fraction)
        lookback_period : int
            Lookback period for calculating statistics
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.transaction_cost = transaction_cost
        self.lookback_period = lookback_period
    
    def generate_signals(self, spread: pd.Series) -> pd.Series:
        """
        Generate trading signals from spread series.
        
        Parameters:
        -----------
        spread : pd.Series
            Spread series
            
        Returns:
        --------
        pd.Series
            Trading signals: 1 for long spread, -1 for short spread, 0 for no position
        """
        signals = pd.Series(0, index=spread.index)
        position = 0
        
        for i in range(self.lookback_period, len(spread)):
            # Calculate rolling statistics
            spread_window = spread.iloc[i - self.lookback_period:i]
            mean_spread = spread_window.mean()
            std_spread = spread_window.std()
            
            if std_spread == 0:
                signals.iloc[i] = position
                continue
            
            # Normalize current spread
            z = (spread.iloc[i] - mean_spread) / std_spread
            
            if position == 0:
                # No position: check for entry
                if z > self.entry_threshold:
                    position = -1  # Short spread (expect mean reversion down)
                elif z < -self.entry_threshold:
                    position = 1   # Long spread (expect mean reversion up)
            elif position == 1:
                # Long position: check for exit
                if z > -self.exit_threshold:  # Spread reverted back toward mean
                    position = 0
            elif position == -1:
                # Short position: check for exit
                if z < self.exit_threshold:  # Spread reverted back toward mean
                    position = 0
            
            signals.iloc[i] = position
        
        return signals
    
    def calculate_returns(
        self,
        spread: pd.Series,
        signals: pd.Series,
        weights: np.ndarray,
        prices: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate strategy returns.
        
        Parameters:
        -----------
        spread : pd.Series
            Spread series
        signals : pd.Series
            Trading signals
        weights : np.ndarray
            Cointegrating weights
        prices : pd.DataFrame
            Price data (log prices)
            
        Returns:
        --------
        pd.Series
            Strategy returns
        """
        returns = pd.Series(0.0, index=spread.index)
        prev_signal = 0
        
        # Calculate asset returns
        asset_returns = prices.diff().dropna()
        spread_returns = spread.diff().dropna()
        
        for i in range(1, len(signals)):
            if i >= len(spread_returns):
                continue
                
            signal = signals.iloc[i]
            prev_signal = signals.iloc[i - 1]
            
            # Calculate return from spread reversion
            # When long spread: profit if spread decreases
            # When short spread: profit if spread increases
            if signal != 0:
                spread_return = -spread_returns.iloc[i] * signal
                
                # Apply transaction cost when position changes
                if signal != prev_signal and prev_signal != 0:
                    spread_return -= self.transaction_cost * 2  # Exit + enter
                elif signal != prev_signal:
                    spread_return -= self.transaction_cost  # Enter only
                
                returns.iloc[i] = spread_return
            else:
                # Exit transaction cost if closing position
                if prev_signal != 0:
                    returns.iloc[i] = -self.transaction_cost
        
        return returns
    
    def backtest(
        self,
        spread: pd.Series,
        weights: np.ndarray,
        prices: pd.DataFrame
    ) -> dict:
        """
        Run backtest and return performance metrics.
        
        Parameters:
        -----------
        spread : pd.Series
            Spread series
        weights : np.ndarray
            Cointegrating weights
        prices : pd.DataFrame
            Price data
            
        Returns:
        --------
        dict
            Dictionary with performance metrics
        """
        from src.utils import (
            sharpe_ratio, max_drawdown, profit_factor, half_life
        )
        
        signals = self.generate_signals(spread)
        returns = self.calculate_returns(spread, signals, weights, prices)
        
        # Calculate metrics
        cumulative_returns = (1 + returns).cumprod()
        
        metrics = {
            'total_return': cumulative_returns.iloc[-1] - 1,
            'sharpe_ratio': sharpe_ratio(returns),
            'max_drawdown': max_drawdown(cumulative_returns),
            'profit_factor': profit_factor(returns),
            'half_life': half_life(spread),
            'num_trades': (signals.diff() != 0).sum(),
            'win_rate': (returns > 0).sum() / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
        }
        
        return metrics
