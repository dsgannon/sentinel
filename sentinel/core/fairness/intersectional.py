import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def intersectional_performance(y_true, y_pred, groups, metric_fn=roc_auc_score):
    """
    Compute model performance across all combinations of protected attributes.

    A model can appear fair on each protected attribute individually while
    still systematically disadvantaging people who belong to multiple protected
    groups. This function surfaces those intersectional disparities by computing
    the metric separately for every unique group combination.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels or values.
    y_pred : array-like
        Model predictions or probabilities.
    groups : pd.DataFrame or dict
        DataFrame where each column is a protected attribute
        (e.g. columns: "race", "gender"). One row per observation.
    metric_fn : callable, default=roc_auc_score
        A function with signature f(y_true, y_pred) -> float.

    Returns
    -------
    pd.DataFrame
        One row per group combination with at least 10 observations.
        Columns: one per protected attribute + "metric_value".
        Combinations with fewer than 10 observations are skipped.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    groups = pd.DataFrame(groups).reset_index(drop = True)

    row = []
    for _, comb in groups.drop_duplicate().iterrows():
        mask = np.ones(len(groups), dtype = bool)
        for col in groups.columns:
            mask &= (groups[col] == combo[col]).values
        
        if mask.sum() < 10:
            continue
        score = metric_fn(y_true[mask], y_pred[mask])
        row = combo.to_dict()
        row["metric_value"] = score
        rows.append(row)
    
    return pd.DateFrame(rows)