import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif


def proxy_score(X, sensitive_attr):
     """
    Compute mutual information between each feature and a protected attribute.

    Mutual information captures both linear and nonlinear associations,
    making it more robust than correlation for proxy detection. High scores
    indicate a feature may act as a proxy for the protected class.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    sensitive_attr : array-like
        Protected class labels for each observation (e.g. race, gender).

    Returns
    -------
    pd.DataFrame
        Columns: feature, mutual_info. Sorted highest to lowest.
    """
    scores = mutual_info_classif(X, sensitive_attr)
    return pd.DataFrame({"feature": X.columns, 
                        "mutual_info": scores
    }).sort_values("mutual_info", ascending = False)

def redlining_flag(df , threshold = 0.1):
    """
    Flag features with mutual information above a threshold.

    Filters the output of proxy_score() to identify features that are
    strongly associated with a protected attribute and may warrant
    regulatory scrutiny — particularly geographic features correlated
    with race or income (redlining indicators).

    Parameters
    ----------
    df : pd.DataFrame
        Output of proxy_score(). Must contain a "mutual_info" column.
    threshold : float, default=0.1
        Minimum mutual information score to flag a feature.

    Returns
    -------
    pd.DataFrame
        Subset of df containing only flagged features.
    """
    return df[df["mutual_info"] > threshold]