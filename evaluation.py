"""
Evaluation and comparison framework for basket trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.cointegration import get_johansen_weights, calculate_spread
from src.bayesian_opt import BayesianOptimizer, create_objective_function
from src.strategy import TradingStrategy
from src.utils import (
    sharpe_ratio, max_drawdown, profit_factor, half_life,
    rolling_window_split
)


class StrategyEvaluator:
    """
    Evaluator for comparing Johansen baseline vs BO-optimized strategies.
    """
    
    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        transaction_cost: float = 0.001,
        lookback_period: int = 60
    ):
        """
        Initialize evaluator.
        
        Parameters:
        -----------
        entry_threshold : float
            Entry threshold for trading strategy
        exit_threshold : float
            Exit threshold for trading strategy
        transaction_cost : float
            Transaction cost per trade
        lookback_period : int
            Lookback period for calculating statistics
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.transaction_cost = transaction_cost
        self.lookback_period = lookback_period
    
    def evaluate_strategy(
        self,
        prices: pd.DataFrame,
        weights: np.ndarray,
        train_start: int,
        train_end: int,
        test_start: int,
        test_end: int
    ) -> Dict:
        """
        Evaluate strategy on test period.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data (log prices)
        weights : np.ndarray
            Cointegrating weights
        train_start : int
            Training start index
        train_end : int
            Training end index
        test_start : int
            Test start index
        test_end : int
            Test end index
            
        Returns:
        --------
        Dict
            Dictionary with performance metrics
        """
        # Get test period data
        test_prices = prices.iloc[test_start:test_end]
        test_spread = calculate_spread(test_prices, weights)
        
        # Create strategy
        strategy = TradingStrategy(
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold,
            transaction_cost=self.transaction_cost,
            lookback_period=self.lookback_period
        )
        
        # Run backtest
        metrics = strategy.backtest(test_spread, weights, test_prices)
        
        # Add additional metrics
        signals = strategy.generate_signals(test_spread)
        returns = strategy.calculate_returns(test_spread, signals, weights, test_prices)
        
        if len(returns) > 0:
            cumulative_returns = (1 + returns).cumprod()
            metrics.update({
                'mean_return': returns.mean(),
                'volatility': returns.std(),
                'mean_return_vol_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'final_cumulative_return': cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
            })
        
        return metrics
    
    def rolling_window_evaluation(
        self,
        prices: pd.DataFrame,
        johansen_weights: np.ndarray,
        bo_weights: np.ndarray,
        window_size: int = 252,
        step_size: int = 63
    ) -> Dict:
        """
        Perform rolling window evaluation.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data (log prices)
        johansen_weights : np.ndarray
            Johansen baseline weights
        bo_weights : np.ndarray
            BO-optimized weights
        window_size : int
            Size of test window
        step_size : int
            Step size between windows
            
        Returns:
        --------
        Dict
            Dictionary with evaluation results
        """
        splits = rolling_window_split(prices, window_size, step_size)
        
        johansen_results = []
        bo_results = []
        
        print(f"Running rolling window evaluation with {len(splits)} windows...")
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
            if train_end < self.lookback_period:
                continue
            
            print(f"Window {i+1}/{len(splits)}: Test period {prices.index[test_start].date()} to {prices.index[test_end-1].date()}")
            
            # Evaluate Johansen strategy
            try:
                johansen_metrics = self.evaluate_strategy(
                    prices, johansen_weights, train_start, train_end, test_start, test_end
                )
                johansen_results.append(johansen_metrics)
            except Exception as e:
                print(f"  Error evaluating Johansen strategy: {e}")
                continue
            
            # Evaluate BO strategy
            try:
                bo_metrics = self.evaluate_strategy(
                    prices, bo_weights, train_start, train_end, test_start, test_end
                )
                bo_results.append(bo_metrics)
            except Exception as e:
                print(f"  Error evaluating BO strategy: {e}")
                continue
        
        # Aggregate results
        if len(johansen_results) == 0 or len(bo_results) == 0:
            return {
                'johansen_results': [],
                'bo_results': [],
                'comparison': {}
            }
        
        johansen_df = pd.DataFrame(johansen_results)
        bo_df = pd.DataFrame(bo_results)
        
        # Calculate summary statistics
        comparison = {
            'johansen': {
                'mean_sharpe': johansen_df['sharpe_ratio'].mean(),
                'std_sharpe': johansen_df['sharpe_ratio'].std(),
                'mean_max_drawdown': johansen_df['max_drawdown'].mean(),
                'mean_profit_factor': johansen_df['profit_factor'].mean(),
                'mean_total_return': johansen_df['total_return'].mean(),
                'mean_half_life': johansen_df['half_life'].mean(),
            },
            'bo': {
                'mean_sharpe': bo_df['sharpe_ratio'].mean(),
                'std_sharpe': bo_df['sharpe_ratio'].std(),
                'mean_max_drawdown': bo_df['max_drawdown'].mean(),
                'mean_profit_factor': bo_df['profit_factor'].mean(),
                'mean_total_return': bo_df['total_return'].mean(),
                'mean_half_life': bo_df['half_life'].mean(),
            },
            'improvement': {
                'sharpe_improvement': (bo_df['sharpe_ratio'].mean() - johansen_df['sharpe_ratio'].mean()) / abs(johansen_df['sharpe_ratio'].mean()) if johansen_df['sharpe_ratio'].mean() != 0 else 0,
                'drawdown_reduction': (johansen_df['max_drawdown'].mean() - bo_df['max_drawdown'].mean()) / johansen_df['max_drawdown'].mean() if johansen_df['max_drawdown'].mean() != 0 else 0,
                'return_improvement': (bo_df['total_return'].mean() - johansen_df['total_return'].mean()) / abs(johansen_df['total_return'].mean()) if johansen_df['total_return'].mean() != 0 else 0,
            }
        }
        
        return {
            'johansen_results': johansen_results,
            'bo_results': bo_results,
            'johansen_df': johansen_df,
            'bo_df': bo_df,
            'comparison': comparison
        }


def compare_strategies(
    prices: pd.DataFrame,
    johansen_weights: np.ndarray,
    bo_weights: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    transaction_cost: float = 0.001,
    window_size: int = 252,
    step_size: int = 63
) -> Dict:
    """
    Compare Johansen baseline vs BO-optimized strategies.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data (log prices)
    johansen_weights : np.ndarray
        Johansen baseline weights
    bo_weights : np.ndarray
        BO-optimized weights
    entry_threshold : float
        Entry threshold
    exit_threshold : float
        Exit threshold
    transaction_cost : float
        Transaction cost
    window_size : int
        Test window size
    step_size : int
        Step size between windows
        
    Returns:
    --------
    Dict
        Dictionary with comparison results
    """
    evaluator = StrategyEvaluator(
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        transaction_cost=transaction_cost
    )
    
    results = evaluator.rolling_window_evaluation(
        prices, johansen_weights, bo_weights, window_size, step_size
    )
    
    return results

