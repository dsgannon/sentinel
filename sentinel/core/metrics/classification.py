import numpy as np
from sklearn.metrics import roc_auc_score


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

