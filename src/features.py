"""
Feature engineering utilities for portfolio optimization.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def calculate_daily_returns(prices):
    """
    Calculate daily percentage returns.
    
    Args:
        prices (pd.Series or pd.DataFrame): Price data
    
    Returns:
        pd.Series or pd.DataFrame: Daily returns
    """
    if prices is None or (hasattr(prices, 'empty') and prices.empty):
        raise ValueError("Input prices cannot be empty or None")

    returns = prices.pct_change().dropna()
    return returns


def calculate_rolling_stats(prices, window=30):
    """
    Calculate rolling mean and standard deviation.
    
    Args:
        prices (pd.Series): Price series
        window (int): Rolling window size in days
    
    Returns:
        tuple: (rolling_mean, rolling_std)
    """
    if window < 1:
        raise ValueError("Window must be at least 1")

    if len(prices) < window:
        raise ValueError(f"Price series too short for window size {window}")

    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()

    return rolling_mean, rolling_std


def calculate_sharpe_ratio(returns, risk_free_rate=0.02, trading_days=252):
    """
    Calculate annualized Sharpe Ratio.
    
    Args:
        returns (pd.Series): Daily returns
        risk_free_rate (float): Annual risk-free rate (default 2%)
        trading_days (int): Trading days per year (default 252)
    
    Returns:
        float: Sharpe Ratio
    """
    if returns is None or len(returns) == 0:
        raise ValueError("Returns series cannot be empty")

    ann_return = returns.mean() * trading_days
    ann_vol = returns.std() * np.sqrt(trading_days)

    if ann_vol == 0:
        raise ValueError("Volatility is zero, cannot calculate Sharpe Ratio")

    sharpe = (ann_return - risk_free_rate) / ann_vol
    return sharpe


def calculate_var(returns, confidence=0.95):
    """
    Calculate Value at Risk at given confidence level.
    
    Args:
        returns (pd.Series): Daily returns
        confidence (float): Confidence level (default 0.95)
    
    Returns:
        float: VaR value
    """
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")

    var = np.percentile(returns, (1 - confidence) * 100)
    return var


def calculate_max_drawdown(returns):
    """
    Calculate maximum drawdown from a returns series.
    
    Args:
        returns (pd.Series): Daily returns
    
    Returns:
        float: Maximum drawdown (negative number)
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()


def adf_test(series):
    """
    Perform Augmented Dickey-Fuller stationarity test.
    
    Args:
        series (pd.Series): Time series to test
    
    Returns:
        dict: Test results with statistic, p-value, and is_stationary flag
    """
    result = adfuller(series.dropna())
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }


def create_lstm_sequences(data, window_size=60):
    """
    Create input sequences for LSTM model.
    
    Args:
        data (np.array): Scaled price data
        window_size (int): Number of past days to use as input
    
    Returns:
        tuple: (X, y) arrays for model training
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")

    if len(data) <= window_size:
        raise ValueError(
            f"Data length ({len(data)}) must be greater than window_size ({window_size})"
        )

    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])

    return np.array(X), np.array(y)