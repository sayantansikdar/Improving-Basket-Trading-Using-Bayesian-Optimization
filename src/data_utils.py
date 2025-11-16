"""
Data fetching and preprocessing utilities.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def fetch_price_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str = '1d'
) -> pd.DataFrame:
    """
    Fetch price data for a list of symbols.
    
    Parameters:
    -----------
    symbols : List[str]
        List of stock/ETF symbols
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    interval : str
        Data interval (default: '1d' for daily)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with prices, indexed by date
    """
    prices = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if len(data) == 0:
                print(f"Warning: No data for {symbol}")
                continue
                
            prices[symbol] = data['Close']
            print(f"Fetched {len(data)} records for {symbol}")
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue
    
    if len(prices) == 0:
        raise ValueError("No data was successfully fetched")
    
    prices_df = pd.DataFrame(prices)
    prices_df.index = pd.to_datetime(prices_df.index)
    
    # Forward fill missing values, then backward fill
    prices_df = prices_df.ffill().bfill()
    
    # Drop rows with any remaining NaN values
    prices_df = prices_df.dropna()
    
    return prices_df


def load_price_data(filepath: str) -> pd.DataFrame:
    """
    Load price data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with prices, indexed by date
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def prepare_data(
    prices: pd.DataFrame,
    min_periods: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare price data for cointegration analysis.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with price data
    min_periods : int
        Minimum number of periods required
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        (log_prices, returns) tuple
    """
    if len(prices) < min_periods:
        raise ValueError(f"Need at least {min_periods} periods, got {len(prices)}")
    
    # Calculate log prices
    log_prices = np.log(prices)
    
    # Calculate returns
    returns = log_prices.diff().dropna()
    
    return log_prices, returns


def standardize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize data to have zero mean and unit variance.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
        
    Returns:
    --------
    pd.DataFrame
        Standardized data
    """
    return (data - data.mean()) / data.std()


def create_spread(
    prices: pd.DataFrame,
    weights: np.ndarray
) -> pd.Series:
    """
    Create a spread series from prices and weights.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data
    weights : np.ndarray
        Cointegrating weights
        
    Returns:
    --------
    pd.Series
        Spread series
    """
    if len(weights) != prices.shape[1]:
        raise ValueError(f"Weights length ({len(weights)}) must match number of assets ({prices.shape[1]})")
    
    log_prices = np.log(prices)
    spread = (log_prices.values @ weights).flatten()
    spread_series = pd.Series(spread, index=prices.index)
    
    return spread_series
