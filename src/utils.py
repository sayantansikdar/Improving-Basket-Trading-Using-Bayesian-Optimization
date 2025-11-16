"""
Utility functions for basket trading analysis.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log returns from prices.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with price data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with log returns
    """
    return np.log(prices / prices.shift(1)).dropna()


def z_score(series: pd.Series) -> pd.Series:
    """
    Calculate z-score of a series.
    
    Parameters:
    -----------
    series : pd.Series
        Input series
        
    Returns:
    --------
    pd.Series
        Z-score normalized series
    """
    return (series - series.mean()) / series.std()


def half_life(spread: pd.Series) -> float:
    """
    Calculate half-life of mean reversion for a spread series.
    
    Parameters:
    -----------
    spread : pd.Series
        Spread series
        
    Returns:
    --------
    float
        Half-life in periods
    """
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    
    spread_lag = spread_lag.iloc[1:]
    spread_diff = spread_diff.iloc[1:]
    
    if len(spread_lag) == 0 or len(spread_diff) == 0:
        return np.nan
    
    # Remove any remaining NaN values
    valid_mask = ~(spread_lag.isna() | spread_diff.isna())
    spread_lag = spread_lag[valid_mask]
    spread_diff = spread_diff[valid_mask]
    
    if len(spread_lag) < 2:
        return np.nan
    
    # OLS regression: spread_diff = alpha + beta * spread_lag
    try:
        theta = np.polyfit(spread_lag, spread_diff, 1)[0]
        if theta >= 0:
            return np.inf
        half_life = -np.log(2) / theta
        return half_life
    except:
        return np.nan


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    periods_per_year : int
        Number of periods per year (default: 252 for daily)
        
    Returns:
    --------
    float
        Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - 0.0  # Assuming risk-free rate = 0
    if excess_returns.std() == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Parameters:
    -----------
    cumulative_returns : pd.Series
        Cumulative return series
        
    Returns:
    --------
    float
        Maximum drawdown (as positive number)
    """
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return abs(drawdown.min())


def profit_factor(returns: pd.Series) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
        
    Returns:
    --------
    float
        Profit factor
    """
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def rolling_window_split(
    data: pd.DataFrame,
    window_size: int,
    step_size: int = 1
) -> list:
    """
    Generate rolling window splits for time series data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    window_size : int
        Size of each window
    step_size : int
        Step size between windows
        
    Returns:
    --------
    list
        List of (train_start, train_end, test_start, test_end) tuples
    """
    splits = []
    n = len(data)
    
    for i in range(window_size, n - window_size, step_size):
        train_start = 0
        train_end = i
        test_start = i
        test_end = min(i + window_size, n)
        splits.append((train_start, train_end, test_start, test_end))
    
    return splits
