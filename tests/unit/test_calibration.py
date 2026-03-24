import numpy as np
from sentinel.core.metrics.calibration import brier_score, expected_calibration_error, reliability_diagram_data

def test_brier_score_perfect():
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    y_pred = np.array([0.0, 0.0, 1.0, 1.0])
    result = brier_score(y_true, y_pred)
    assert result == 0.0

def test_ece_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    result = expected_calibration_error(y_true, y_pred)
    assert result >= 0.0 and result <= 1.0

def test_reliability_diagram_data_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    result = reliability_diagram_data(y_true, y_pred)
    assert len(result[0]) == len(result[1])