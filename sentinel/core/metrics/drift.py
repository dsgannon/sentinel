import numpy as np
import pandas as pd
import warnings

def psi(expected, actual, n_bins=10):
    """
    Compute Population Stability Index (PSI).

    Measures how much a population has shifted between a reference
    (training) dataset and a current (scoring) dataset. Used in
    SR 11-7 validation to detect when a model may need revalidation.

    Parameters
    ----------
    expected : array-like of shape (n_samples,)
        Predicted probabilities or scores from the reference population
        (typically training or validation data).
    actual : array-like of shape (n_samples,)
        Predicted probabilities or scores from the current population
        (typically recent production scoring data).
    n_bins : int, default 10
        Number of bins to use for the histogram comparison.

    Returns
    -------
    float
        PSI value. Interpretation:
        < 0.10  — no significant shift, model is stable.
        0.10 to 0.25 — moderate shift, monitor closely.
        > 0.25  — major shift, model needs revalidation.
    """

    breakpoints = np.linspace(0, 1, n_bins + 1)

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct   = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # avoid division by zero or log(0)
    expected_pct = np.clip(expected_pct , 1e-6, None)
    actual_pct   = np.clip(actual_pct,    1e-6, None)

    # PSI Formula
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    if psi_value > 0.25:
        warnings.warn(f"PSI={psi_value:.3f} exceeds 0.25 — major population shift detected.")
    return psi_value

def csi(expected_df, actual_df, n_bins=10):
    """
    Compute Characteristic Stability Index (CSI) for each feature.

    Runs PSI on every column in the DataFrame to identify which
    features are driving population shift. Useful for diagnosing
    high PSI scores at the feature level.

    Parameters
    ----------
    expected_df : pd.DataFrame
        Feature values from the reference population (training data).
    actual_df : pd.DataFrame
        Feature values from the current scoring population.
    n_bins : int, default 10
        Number of bins to use for each feature's PSI calculation.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: feature, csi, flag.
        flag is 'stable' (<0.10), 'monitor' (0.10-0.25),
        or 'unstable' (>0.25).
    """
    result = []
    for col in expected_df.columns:
        csi_value = psi(expected_df[col], actual_df[col], n_bins)
        if csi_value < 0.10:
            flag = 'stable'
        elif csi_value < 0.25:
            flag = 'monitor'
        else:
            flag = 'unstable'
        result.append({
            'feature': col,
            'csi': round(csi_value, 4),
            'flag': flag
        })
    return pd.DataFrame(result)