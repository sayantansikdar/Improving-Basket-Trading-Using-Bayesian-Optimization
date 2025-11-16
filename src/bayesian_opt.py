"""
Bayesian Optimization for finding optimal cointegrating weights.
"""

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from typing import Callable, Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class BayesianOptimizer:
    """
    Bayesian Optimization framework for optimizing basket trading weights.
    """
    
    def __init__(
        self,
        objective_func: Callable,
        n_assets: int,
        n_trials: int = 50,
        n_random_starts: int = 10,
        normalize_weights: bool = True
    ):
        """
        Initialize Bayesian Optimizer.
        
        Parameters:
        -----------
        objective_func : Callable
            Objective function to minimize (should return negative performance metric)
        n_assets : int
            Number of assets in the basket
        n_trials : int
            Number of optimization trials
        n_random_starts : int
            Number of random starting points
        normalize_weights : bool
            Whether to normalize weights so they sum to a constraint
        """
        self.objective_func = objective_func
        self.n_assets = n_assets
        self.n_trials = n_trials
        self.n_random_starts = n_random_starts
        self.normalize_weights = normalize_weights
        
        # Define search space: weights for each asset (last one will be determined by normalization)
        # We search for n_assets - 1 weights, the last is constrained
        self.dimensions = [
            Real(-2.0, 2.0, name=f'weight_{i}') 
            for i in range(n_assets - 1)
        ]
        
        # Store optimization results
        self.result = None
        self.best_weights = None
        self.best_score = None
    
    def normalize_weights_array(self, weights: np.ndarray) -> np.ndarray:
        """
        Normalize weights array.
        
        Parameters:
        -----------
        weights : np.ndarray
            Raw weights (may have n_assets - 1 or n_assets elements)
            
        Returns:
        --------
        np.ndarray
            Normalized weights array of length n_assets
        """
        if len(weights) == self.n_assets - 1:
            # Last weight is determined by normalization constraint
            # For cointegration, typically sum to zero (long-short neutral)
            last_weight = -np.sum(weights)
            weights_full = np.append(weights, last_weight)
        else:
            weights_full = weights.copy()
        
        if self.normalize_weights:
            # Normalize so last weight is -1 (standard cointegration form)
            if weights_full[-1] != 0:
                weights_full = weights_full / abs(weights_full[-1])
                if weights_full[-1] > 0:
                    weights_full = -weights_full
        
        return weights_full
    
    @property
    def _objective_wrapper(self):
        """Wrapper for objective function to handle weight normalization."""
        @use_named_args(dimensions=self.dimensions)
        def objective(**params):
            # Extract weights in order
            weights_list = [params[f'weight_{i}'] for i in range(self.n_assets - 1)]
            weights = np.array(weights_list)
            
            # Normalize weights
            weights_normalized = self.normalize_weights_array(weights)
            
            # Call objective function
            score = self.objective_func(weights_normalized)
            
            # Return negative score (since we minimize)
            return -score
        
        return objective
    
    def optimize(self, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Run Bayesian optimization.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print optimization progress
            
        Returns:
        --------
        Tuple[np.ndarray, float]
            (best_weights, best_score) tuple
        """
        objective = self._objective_wrapper
        
        # Run optimization
        self.result = gp_minimize(
            func=objective,
            dimensions=self.dimensions,
            n_calls=self.n_trials,
            n_initial_points=self.n_random_starts,
            acq_func='EI',  # Expected Improvement
            n_jobs=1,
            random_state=42,
            verbose=verbose
        )
        
        # Extract best weights
        best_weights_raw = self.result.x
        self.best_weights = self.normalize_weights_array(np.array(best_weights_raw))
        self.best_score = -self.result.fun  # Negate back to positive
        
        return self.best_weights, self.best_score
    
    def get_optimization_history(self) -> Dict:
        """
        Get optimization history.
        
        Returns:
        --------
        Dict
            Dictionary with optimization history
        """
        if self.result is None:
            return {}
        
        return {
            'x_iters': self.result.x_iters,
            'func_vals': [-x for x in self.result.func_vals],  # Negate back
            'iterations': list(range(len(self.result.func_vals)))
        }


def create_objective_function(
    prices: pd.DataFrame,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    transaction_cost: float = 0.001,
    metric: str = 'sharpe'
) -> Callable:
    """
    Create objective function for Bayesian optimization.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data (log prices)
    entry_threshold : float
        Entry threshold (standard deviations)
    exit_threshold : float
        Exit threshold (standard deviations)
    transaction_cost : float
        Transaction cost per trade (as fraction)
    metric : str
        Performance metric to optimize: 'sharpe', 'return', 'profit_factor'
        
    Returns:
    --------
    Callable
        Objective function that takes weights and returns performance score
    """
    from src.strategy import TradingStrategy
    from src.utils import sharpe_ratio
    
    def objective(weights: np.ndarray) -> float:
        """
        Objective function to maximize.
        
        Parameters:
        -----------
        weights : np.ndarray
            Cointegrating weights
            
        Returns:
        --------
        float
            Performance metric (to maximize)
        """
        try:
            # Calculate spread
            spread = (prices.values @ weights).flatten()
            spread_series = pd.Series(spread, index=prices.index)
            
            # Create strategy
            strategy = TradingStrategy(
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                transaction_cost=transaction_cost
            )
            
            # Generate signals and calculate returns
            signals = strategy.generate_signals(spread_series)
            returns = strategy.calculate_returns(spread_series, signals, weights, prices)
            
            # Calculate performance metric
            if metric == 'sharpe':
                if len(returns) == 0 or returns.std() == 0:
                    return -1e6
                score = sharpe_ratio(returns)
            elif metric == 'return':
                score = returns.mean() * 252  # Annualized
            elif metric == 'profit_factor':
                gross_profit = returns[returns > 0].sum()
                gross_loss = abs(returns[returns < 0].sum())
                if gross_loss == 0:
                    score = gross_profit if gross_profit > 0 else 0.0
                else:
                    score = gross_profit / gross_loss
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            # Return negative if score is NaN or inf
            if np.isnan(score) or np.isinf(score):
                return -1e6
            
            return float(score)
            
        except Exception as e:
            # Return very negative value on error
            return -1e6
    
    return objective


def optimize_weights_multi_objective(
    prices: pd.DataFrame,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    transaction_cost: float = 0.001,
    n_trials: int = 50,
    weights: Optional[List[float]] = None
) -> Dict:
    """
    Optimize weights for multiple objectives using weighted sum approach.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data
    entry_threshold : float
        Entry threshold
    exit_threshold : float
        Exit threshold
    transaction_cost : float
        Transaction cost
    n_trials : int
        Number of trials
    weights : Optional[List[float]]
        Weights for different objectives [sharpe_weight, return_weight, pf_weight]
        
    Returns:
    --------
    Dict
        Dictionary with optimization results
    """
    if weights is None:
        weights = [0.5, 0.3, 0.2]  # Default: 50% Sharpe, 30% Return, 20% Profit Factor
    
    # Create multi-objective function
    def multi_objective(w: np.ndarray) -> float:
        try:
            spread = (prices.values @ w).flatten()
            spread_series = pd.Series(spread, index=prices.index)
            
            from src.strategy import TradingStrategy
            from src.utils import sharpe_ratio, profit_factor
            
            strategy = TradingStrategy(
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                transaction_cost=transaction_cost
            )
            
            signals = strategy.generate_signals(spread_series)
            returns = strategy.calculate_returns(spread_series, signals, w, prices)
            
            if len(returns) == 0:
                return -1e6
            
            sharpe = sharpe_ratio(returns)
            annual_return = returns.mean() * 252
            pf = profit_factor(returns)
            
            # Normalize metrics (rough normalization)
            sharpe_norm = sharpe / 3.0  # Assume max Sharpe ~3
            return_norm = annual_return / 0.5  # Assume max return ~50%
            pf_norm = pf / 5.0  # Assume max PF ~5
            
            score = (weights[0] * sharpe_norm + 
                    weights[1] * return_norm + 
                    weights[2] * pf_norm)
            
            if np.isnan(score) or np.isinf(score):
                return -1e6
            
            return float(score)
            
        except:
            return -1e6
    
    # Optimize
    optimizer = BayesianOptimizer(
        objective_func=multi_objective,
        n_assets=prices.shape[1],
        n_trials=n_trials
    )
    
    best_weights, best_score = optimizer.optimize()
    
    return {
        'best_weights': best_weights,
        'best_score': best_score,
        'optimization_history': optimizer.get_optimization_history()
    }
