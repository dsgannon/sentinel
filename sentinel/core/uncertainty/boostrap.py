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

    