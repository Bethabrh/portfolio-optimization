"""
Data loading and validation utilities for portfolio optimization.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime


def download_stock_data(tickers, start_date, end_date):
    """
    Download historical stock data from Yahoo Finance.
    
    Args:
        tickers (list): List of ticker symbols e.g. ['TSLA', 'SPY', 'BND']
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        pd.DataFrame: Closing prices for all tickers
    
    Raises:
        ValueError: If tickers list is empty or dates are invalid
        RuntimeError: If download fails
    """
    if not tickers:
        raise ValueError("Tickers list cannot be empty")

    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Dates must be in YYYY-MM-DD format")

    if start_date >= end_date:
        raise ValueError("start_date must be before end_date")

    try:
        raw_data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download data from Yahoo Finance: {e}")

    if raw_data.empty:
        raise RuntimeError("Downloaded data is empty. Check tickers and date range.")

    # Extract closing prices
    if isinstance(tickers, list) and len(tickers) > 1:
        close_prices = raw_data['Close']
    else:
        close_prices = raw_data[['Close']]
        close_prices.columns = tickers

    return close_prices


def clean_data(df):
    """
    Clean and validate a DataFrame of closing prices.
    
    Args:
        df (pd.DataFrame): Raw price DataFrame
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    
    Raises:
        ValueError: If DataFrame is empty after cleaning
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")

    # Forward fill then backward fill missing values
    df_clean = df.ffill().bfill()

    # Drop any remaining NaN rows
    df_clean = df_clean.dropna()

    if df_clean.empty:
        raise ValueError("DataFrame is empty after cleaning")

    # Ensure all values are positive (prices can't be negative)
    if (df_clean < 0).any().any():
        raise ValueError("Negative prices detected in data")

    return df_clean


def validate_data(df, expected_tickers):
    """
    Validate that downloaded data contains expected tickers and sufficient rows.
    
    Args:
        df (pd.DataFrame): Price DataFrame
        expected_tickers (list): Expected column names
    
    Returns:
        bool: True if valid
    
    Raises:
        ValueError: If validation fails
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty")

    missing = [t for t in expected_tickers if t not in df.columns]
    if missing:
        raise ValueError(f"Missing tickers in data: {missing}")

    if len(df) < 100:
        raise ValueError(f"Insufficient data: only {len(df)} rows (minimum 100 required)")

    return True


def load_processed_data(filepath):
    """
    Load processed data from a CSV file.
    
    Args:
        filepath (str): Path to CSV file
    
    Returns:
        pd.DataFrame: Loaded DataFrame with DatetimeIndex
    
    Raises:
        FileNotFoundError: If file does not exist
    """
    try:
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {filepath}: {e}")

    return df