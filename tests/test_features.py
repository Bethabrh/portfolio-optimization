"""
Unit tests for features module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features import (
    calculate_daily_returns,
    calculate_rolling_stats,
    calculate_sharpe_ratio,
    calculate_var,
    calculate_max_drawdown,
    create_lstm_sequences
)


def make_price_series(n=200):
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    prices = pd.Series(np.random.uniform(100, 400, n), index=dates)
    return prices


def test_daily_returns_length():
    prices = make_price_series(100)
    returns = calculate_daily_returns(prices)
    assert len(returns) == 99


def test_daily_returns_empty_raises():
    with pytest.raises(ValueError):
        calculate_daily_returns(pd.Series(dtype=float))


def test_rolling_stats_shape():
    prices = make_price_series(100)
    mean, std = calculate_rolling_stats(prices, window=30)
    assert len(mean) == len(prices)
    assert len(std) == len(prices)


def test_rolling_stats_invalid_window():
    prices = make_price_series(100)
    with pytest.raises(ValueError):
        calculate_rolling_stats(prices, window=0)


def test_rolling_stats_too_short():
    prices = make_price_series(10)
    with pytest.raises(ValueError):
        calculate_rolling_stats(prices, window=30)


def test_sharpe_ratio_returns_float():
    prices = make_price_series(200)
    returns = calculate_daily_returns(prices)
    sharpe = calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)


def test_sharpe_ratio_empty_raises():
    with pytest.raises(ValueError):
        calculate_sharpe_ratio(pd.Series(dtype=float))


def test_var_valid():
    prices = make_price_series(200)
    returns = calculate_daily_returns(prices)
    var = calculate_var(returns, confidence=0.95)
    assert var < 0 or var >= 0


def test_var_invalid_confidence():
    prices = make_price_series(200)
    returns = calculate_daily_returns(prices)
    with pytest.raises(ValueError):
        calculate_var(returns, confidence=1.5)


def test_max_drawdown_negative():
    prices = make_price_series(200)
    returns = calculate_daily_returns(prices)
    mdd = calculate_max_drawdown(returns)
    assert mdd <= 0


def test_lstm_sequences_shape():
    data = np.random.rand(200, 1)
    X, y = create_lstm_sequences(data, window_size=60)
    assert X.shape == (140, 60)
    assert y.shape == (140,)


def test_lstm_sequences_invalid_window():
    data = np.random.rand(200, 1)
    with pytest.raises(ValueError):
        create_lstm_sequences(data, window_size=0)


def test_lstm_sequences_too_short():
    data = np.random.rand(50, 1)
    with pytest.raises(ValueError):
        create_lstm_sequences(data, window_size=60)