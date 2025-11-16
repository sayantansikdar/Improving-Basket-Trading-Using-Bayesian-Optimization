"""
Cointegration analysis using Johansen test.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from typing import Tuple, Optional


def johansen_test(
    data: pd.DataFrame,
    det_order: int = -1,
    k_ar_diff: int = 1
) -> Tuple[float, float, np.ndarray, Optional[np.ndarray]]:
    """
    Perform Johansen cointegration test.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data (log prices recommended)
    det_order : int
        Deterministic term order:
        -1: no deterministic terms
        0: constant term
        1: linear trend
    k_ar_diff : int
        Number of lags in VAR model
        
    Returns:
    --------
    Tuple containing:
    - trace_stat : float
        Trace test statistic
    - eigen_stat : float
        Eigenvalue test statistic
    - eigenvectors : np.ndarray
        Cointegrating vectors (normalized)
    - evec : np.ndarray
        Full eigenvector matrix (optional)
    """
    # Prepare data (must be numpy array)
    y = data.values
    
    # Perform Johansen test
    result = coint_johansen(y, det_order, k_ar_diff)
    
    # Extract statistics
    trace_stat = result.lr1[0]  # Trace statistic for first cointegrating vector
    eigen_stat = result.lr2[0]  # Eigenvalue statistic for first cointegrating vector
    
    # Extract cointegrating vectors (eigenvectors)
    # The first eigenvector is the most significant cointegrating relationship
    if len(result.evec) > 0:
        eigenvectors = result.evec[:, 0]
        
        # Normalize so last weight is -1 (standard form)
        if eigenvectors[-1] != 0:
            eigenvectors = eigenvectors / abs(eigenvectors[-1])
            if eigenvectors[-1] > 0:
                eigenvectors = -eigenvectors
        
        return trace_stat, eigen_stat, eigenvectors, result.evec
    else:
        return trace_stat, eigen_stat, np.array([]), None


def calculate_spread(
    prices: pd.DataFrame,
    weights: np.ndarray
) -> pd.Series:
    """
    Calculate spread using cointegrating weights.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data (log prices)
    weights : np.ndarray
        Cointegrating weights
        
    Returns:
    --------
    pd.Series
        Spread series
    """
    if len(weights) != prices.shape[1]:
        raise ValueError(f"Weights length ({len(weights)}) must match number of assets ({prices.shape[1]})")
    
    spread = (prices.values @ weights).flatten()
    spread_series = pd.Series(spread, index=prices.index)
    
    return spread_series


def check_cointegration(
    spread: pd.Series,
    critical_value: float = 3.0
) -> Tuple[bool, float]:
    """
    Check if spread is stationary using ADF test approximation.
    
    Parameters:
    -----------
    spread : pd.Series
        Spread series
    critical_value : float
        Critical value for stationarity (rough approximation)
        
    Returns:
    --------
    Tuple[bool, float]
        (is_stationary, adf_statistic)
    """
    from statsmodels.tsa.stattools import adfuller
    
    try:
        adf_result = adfuller(spread.dropna())
        adf_stat = adf_result[0]
        is_stationary = adf_stat < -critical_value
        return is_stationary, adf_stat
    except:
        return False, np.nan


def get_johansen_weights(
    prices: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1
) -> Tuple[np.ndarray, dict]:
    """
    Get cointegrating weights using Johansen test.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Log price data
    det_order : int
        Deterministic term order
    k_ar_diff : int
        Number of lags
        
    Returns:
    --------
    Tuple[np.ndarray, dict]
        (weights, statistics) tuple
    """
    trace_stat, eigen_stat, eigenvectors, evec = johansen_test(
        prices, det_order=det_order, k_ar_diff=k_ar_diff
    )
    
    stats = {
        'trace_statistic': trace_stat,
        'eigenvalue_statistic': eigen_stat,
        'eigenvectors': evec
    }
    
    return eigenvectors, stats
