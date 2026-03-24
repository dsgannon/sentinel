import numpy as np
from sentinel.core.metrics.regression import mae, rmse, r2, mape

def test_mae_perfect():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    result = mae(y_true, y_pred)
    assert result == 0.0


def test_rmse_perfect():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    result = rmse(y_true, y_pred)
    assert result == 0.0


def test_mape_perfect():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    result = mape(y_true, y_pred)
    assert result == 0.0


def test_r2_perfect():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    result = r2(y_true, y_pred)
    assert result == 1.0