import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

def roc_auc(y_true, y_prob):
    """
    Compute Area Under the ROC Curve

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        AUC-ROC score between 0 and 1.
    """
    return roc_auc_score(y_true, y_prob)

def roc_auc_with_ci(y_true, y_prob, n_bootstrap=1000):
    """
    Computes AUC-ROC with the boostrap confidence interval

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    n_bootstrap : int, default 1000
        Number of bootstrap resamples.

    Returns
    -------
    auc : float                                                                         
        AUC-ROC score on the full dataset.                                              
    lower : float                                                                       
        Lower bound of the 95% confidence interval.
    upper : float                                                                       
        Upper bound of the 95% confidence interval.
                                                 
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        # randomly sample n indicies WITH replacement
        indices = np.random.choice(n, size = n, replace = True)
        # calculate AUC on this sample and append to scores
        scores.append(roc_auc(y_true[indices], y_prob[indices]))
    auc = roc_auc(y_true, y_prob)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return auc, lower, upper


def gini(y_true, y_prob):
    """
    Compute Gini Coefficient

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Gini coeficient between 0 and 1.
    """
    return 2 * roc_auc(y_true, y_prob) - 1 

def ks_statistic(y_true, y_prob):
    """
    Compute the Kolmogorov-Smirnov statistic.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.

    Returns
    -------
    statistic : float                                                                         
        The maximum distance between the cumulative distribution of predicted probabilities
        for positives versus negatives. A value between 0 and 1 where higher means better
        separation between the two classes.                                              
    pvalue : float                                                                       
        The probability of observing this much separation by chance if the model had no real
        discriminatory power. A small p-value (typically < 0.05) means the separation is
        statistically significant — the model is genuinely distinguishing between classes,
        not just getting lucky.
    """
    pos = y_prob[y_true == 1]    
    neg = y_prob[y_true == 0]
    result = ks_2samp(pos, neg)
    return result.statistic, result.pvalue

def lift_table(y_true, y_prob, n_bins=10):
    """
    Compute a decile lift table for a binary classification model.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    n_bins : int, default 10
        Number of equal-sized bins to split predictions into.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: decile, n, n_pos, pos_rate,
        cumulative_pos_pct, lift. Sorted from highest to lowest
        predicted probability.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # sort by predicted probability highest to lowest
    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # overall positive rate
    overall_pos_rate = y_true.sum()/len(y_true)

    # split into deciles
    deciles = np.array_split(y_true_sorted, n_bins)

    rows = []
    total_pos = y_true.sum()
    cumulative_pos = 0

    for i, decile in enumerate(deciles):
        n = len(decile)
        n_pos = decile.sum()
        pos_rate = n_pos/n
        cumulative_pos += n_pos
        cumulative_pos_pct = cumulative_pos/total_pos
        lift = pos_rate / overall_pos_rate
        rows.append({
            'decile': i + 1,
            'n': n,
            'n_pos': n_pos,
            'pos_rate': round(pos_rate, 4),
            'cumulative_pos_pct': round(cumulative_pos_pct, 4),
            'lift': round(lift, 4)
        })

    return pd.DataFrame(rows)
