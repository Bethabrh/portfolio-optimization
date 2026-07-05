"""
Unit tests for data_loader module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import clean_data, validate_data, download_stock_data


def make_sample_df():
    dates = pd.date_range('2020-01-01', periods=200, freq='B')
    return pd.DataFrame({
        'TSLA': np.random.uniform(100, 400, 200),
        'SPY': np.random.uniform(200, 500, 200),
        'BND': np.random.uniform(70, 90, 200),
    }, index=dates)


def test_clean_data_removes_nulls():
    df = make_sample_df()
    df.iloc[5, 0] = np.nan
    df.iloc[10, 1] = np.nan
    result = clean_data(df)
    assert result.isnull().sum().sum() == 0


def test_clean_data_empty_raises():
    with pytest.raises(ValueError):
        clean_data(pd.DataFrame())


def test_clean_data_none_raises():
    with pytest.raises((ValueError, AttributeError)):
        clean_data(None)


def test_validate_data_passes():
    df = make_sample_df()
    assert validate_data(df, ['TSLA', 'SPY', 'BND']) is True


def test_validate_data_missing_ticker():
    df = make_sample_df()
    with pytest.raises(ValueError, match="Missing tickers"):
        validate_data(df, ['TSLA', 'SPY', 'BND', 'AAPL'])


def test_validate_data_insufficient_rows():
    df = make_sample_df().head(50)
    with pytest.raises(ValueError, match="Insufficient data"):
        validate_data(df, ['TSLA', 'SPY', 'BND'])


def test_validate_data_empty_raises():
    with pytest.raises(ValueError):
        validate_data(pd.DataFrame(), ['TSLA'])


def test_download_empty_tickers_raises():
    with pytest.raises(ValueError, match="empty"):
        download_stock_data([], '2020-01-01', '2021-01-01')


def test_download_invalid_dates_raises():
    with pytest.raises(ValueError):
        download_stock_data(['TSLA'], 'invalid-date', '2021-01-01')


def test_download_start_after_end_raises():
    with pytest.raises(ValueError):
        download_stock_data(['TSLA'], '2022-01-01', '2020-01-01')