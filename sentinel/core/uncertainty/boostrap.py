from scipy.stats import norm
import numpy as np
def bootstrap_ci(metric_fn, y_true, y_pred, n_bootstrap=1000, ci=0.95):
    """                                                                                 
    Compute a bootstrap confidence interval for any scalar metric.                      
                                                                                      
    Repeatedly resamples (y_true, y_pred) with replacement and evaluates                
    metric_fn on each resample to build an empirical distribution of the                
    metric, then returns the point estimate and CI bounds.                              
                                                                                      
    Parameters                                                                          
    ----------                                                                          
    metric_fn : callable
        A function with signature f(y_true, y_pred) -> float.
        Examples: roc_auc, mae, brier_score.                                            
    y_true : array-like                                                                 
        Ground truth labels or values.                                                  
    y_pred : array-like                                                                 
        Model predictions or probabilities.                                             
    n_bootstrap : int, default=1000
        Number of bootstrap resamples.
    ci : float, default=0.95                                                            
        Confidence level (e.g. 0.95 → 95% interval).
                                                                                      
    Returns
    -------                                                                             
    tuple of (float, float, float)
        (point_estimate, lower_bound, upper_bound)
    """  
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        # randomly sample n indicies WITH replacement
        indices = np.random.choice(n, size = n, replace = True)
        # calculate AUC on this sample and append to scores
        scores.append(metric_fn(y_true[indices], y_pred[indices]))
    point_estimate = metric_fn(y_true, y_pred)
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1-(1 - ci) / 2) *100)
    return point_estimate, lower, upper

def bootstrap_ci_bca(metric_fn, y_true, y_pred, n_bootstrap=1000, ci=0.95):   
    """
    Compute a bias-corrected and accelerated (BCa) bootstrap confidence interval.

    More accurate than the percentile method when the bootstrap distribution
    is skewed or when the standard error of the metric varies with the parameter
    value. Uses jackknife resampling to estimate the acceleration factor.

    Parameters
    ----------
    metric_fn : callable
        A function with signature f(y_true, y_pred) -> float.
        Examples: roc_auc, mae, brier_score.
    y_true : array-like
        Ground truth labels or values.
    y_pred : array-like
        Model predictions or probabilities.
    n_bootstrap : int, default=1000
        Number of bootstrap resamples.
    ci : float, default=0.95
        Confidence level (e.g. 0.95 → 95% interval).

    Returns
    -------
    tuple of (float, float, float)
        (point_estimate, lower_bound, upper_bound)
        Bounds are bias-corrected and acceleration-adjusted.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        # randomly sample n indicies WITH replacement
        indices = np.random.choice(n, size = n, replace = True)
        # calculate AUC on this sample and append to scores
        scores.append(metric_fn(y_true[indices], y_pred[indices]))
    scores = np.array(scores)
    point_estimate = metric_fn(y_true, y_pred)
    z0 = norm.ppf(np.mean(scores < point_estimate))
    jack_scores = np.array([
        metric_fn(np.delete(y_true, i), np.delete(y_pred, i))
        for i in range(n)
    ])
    jack_mean = np.mean(jack_scores)
    a = np.sum((jack_mean - jack_scores)**3) / (6 * np.sum((jack_mean - jack_scores)**2)**1.5)
    
    z_low = norm.ppf((1 - ci) / 2)
    z_high = norm.ppf(1 - (1-ci) / 2)

    p_low = norm.cdf(z0 + (z0 + z_low) / (1 - a * (z0 + z_low))) * 100
    p_high = norm.cdf(z0 + (z0 + z_high) / (1 - a * (z0 + z_high))) * 100

    lower = np.percentile(scores, p_low)
    upper = np.percentile(scores, p_high)
    return point_estimate, lower, upper