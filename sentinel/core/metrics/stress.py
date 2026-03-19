import numpy as np
import pandas as pd

def feature_sensitivity(X, y_true, predict_fn, metric_fn, noise_std=0.1):
     """
    Measure how much each feature contributes to model performance by perturbing
    it with Gaussian noise and recording the change in the target metric.

    Each feature is perturbed independently — all other features remain unchanged.
    A large metric_delta indicates the model is highly sensitive to that feature,
    which may warrant additional monitoring in production.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y_true : array-like
        Ground truth labels or values.
    predict_fn : callable
        A function that takes X and returns predictions or probabilities.
    metric_fn : callable
        A function with signature f(y_true, y_pred) -> float.
    noise_std : float, default=0.1
        Standard deviation of the Gaussian noise added to each feature.

    Returns
    -------
    pd.DataFrame
        Columns: feature, metric_delta. One row per feature.
        metric_delta is the perturbed metric minus the baseline metric.
    """
    y_true = np.array(y_true)
    baseline = metric_fn(y_true, predict_fn(X))
    rows = []
    X_copy = X.copy()
    for col in X:
        X_copy[col] = X_copy[col] + np.random.normal(0, noise_std, size=len(X))                                         
        predictions = predict_fn(X_copy)
        metrics = metric_fn(predictions, y_true)
        metric_delta = metrics - baseline
        rows.append({"feature": col, "metric_delta": metric_delta})
        X_copy[col] = X[col]
    return pd.DataFrame(rows)


def stress_test(X, y_true, predict_fn, metric_fn, mask):
    """
    Evaluate model performance on a stressed subset of the data.

    Applies a boolean mask to select a challenging or adverse scenario
    (e.g. recession-era records, low-income borrowers, out-of-time data)
    and returns the metric on that subset alone.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    y_true : array-like
        Full ground truth labels or values.
    predict_fn : callable
        A function that takes X and returns predictions or probabilities.
    metric_fn : callable
        A function with signature f(y_true, y_pred) -> float.
    mask : array-like of bool
        Boolean array selecting the stress subset.

    Returns
    -------
    float
        Metric value on the stressed subset.
    """
    y_true = np.array(y_true)
    pred = predict_fn(X[mask])
    return metric_fn(y_true[mask], pred)
