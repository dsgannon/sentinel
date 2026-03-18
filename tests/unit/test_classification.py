import numpy as np
from sentinel.core.metrics.classification import roc_auc, roc_auc_with_ci, gini, ks_statistic
from sentinel.core.metrics.calibration import brier_score

def test_roc_auc_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    result = roc_auc(y_true, y_prob)
    assert result == 1.0

def test_roc_auc_random():
    y_true = np.array([0,1,0,1])
    y_prob = np.array([0.5,0.5,0.5,0.5])
    result = roc_auc(y_true, y_prob)
    assert result == 0.5

def test_gini_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    result = gini(y_true, y_prob)
    assert result == 1.0

def test_ks_statistic_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    ks, pval = ks_statistic(y_true, y_prob)
    assert ks == 1.0
    assert pval < 0.5

def test_brier_score_perfect():
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    result = brier_score(y_true, y_prob)
    assert result == 0.0