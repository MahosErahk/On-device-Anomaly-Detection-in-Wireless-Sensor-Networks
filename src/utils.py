"""Persistence and preprocessing helpers."""

from pathlib import Path
import numpy as np


def save_scaler_params(path, scaler):
    """Save the essential fitted MinMaxScaler parameters to an NPZ file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, data_min=scaler.data_min_, data_max=scaler.data_max_,
             feature_min=scaler.feature_range[0], feature_max=scaler.feature_range[1])


def reconstruction_mse(actual, predicted):
    return np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2, axis=tuple(range(1, np.ndim(actual))))
