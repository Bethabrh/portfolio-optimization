"""
Model training and evaluation utilities for portfolio optimization.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def evaluate_model(actual, predicted, model_name):
    """
    Calculate MAE, RMSE, and MAPE for a model.
    
    Args:
        actual (array-like): Actual values
        predicted (array-like): Predicted values
        model_name (str): Name of the model for display
    
    Returns:
        dict: Dictionary with MAE, RMSE, MAPE
    """
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()

    if len(actual) != len(predicted):
        raise ValueError(
            f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}"
        )

    if len(actual) == 0:
        raise ValueError("Cannot evaluate empty arrays")

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return {
        'Model': model_name,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'MAPE': round(mape, 4)
    }


def scale_data(data):
    """
    Scale data to [0, 1] range using MinMaxScaler.
    
    Args:
        data (array-like): Data to scale
    
    Returns:
        tuple: (scaled_data, scaler) where scaler can be used to inverse transform
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))
    return scaled, scaler


def train_test_split_ts(series, train_end_date):
    """
    Split a time series chronologically into train and test sets.
    
    Args:
        series (pd.Series): Time series with DatetimeIndex
        train_end_date (str): Last date for training set (YYYY-MM-DD)
    
    Returns:
        tuple: (train, test) Series
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")

    train = series[:train_end_date]
    test = series[train_end_date:]
    # Remove overlap
    test = test[test.index > train.index[-1]]

    if len(train) == 0 or len(test) == 0:
        raise ValueError("Train or test set is empty after split")

    return train, test