import numpy as np                                                                  
from scipy.stats import ttest_rel  
from sentinel.core.uncertainty.bootstrap import bootstrap_ci
from sentinel.core.metrics.classification import roc_auc

def compare_regression(y_true, errors_champ, errors_chal, alpha=0.05):
    """                                                                             
    Compare champion and challenger regression models using a paired t-test.        
                                                                                    
    Tests whether the difference in per-observation absolute errors is              
    statistically significant, accounting for the correlation introduced            
    by evaluating both models on the same holdout set.                              
                
    Parameters                                                                      
    ----------  
    y_true : array-like
        Ground truth values (unused in computation, included for API consistency).  
    errors_champ : array-like                                                       
        Absolute errors for the champion model, one per observation.                
    errors_chal : array-like                                                        
        Absolute errors for the challenger model, one per observation.              
    alpha : float, default=0.05                                                     
        Significance level for the hypothesis test.
                                                                                    
    Returns     
    -------
    dict with keys:
        champ_mae, chal_mae : float — mean absolute error for each model.
        t_statistic : float — paired t-test statistic.                              
        p_value : float — two-tailed p-value.                                       
        challenger_wins : bool — True if challenger MAE is lower and p-value < alpha.                                                                              
    """         
    statistic, pvalue = ttest_rel(errors_champ, errors_chal)
    champ_mae = np.mean(errors_champ)
    chal_mae = np.mean(errors_chal) 
    return dict(
                champ_mae = champ_mae, 
                chal_mae = chal_mae, 
                t_statistic = statistic, 
                p_value = pvalue,
                challenger_wins = chal_mae < champ_mae and pvalue < alpha
            )

def compare_auc(y_true, prob_champ, prob_chal):
    """                                                                             
    Compare champion and challenger classifier AUCs using bootstrap confidence intervals.                                                                          
                                                                                    
    A challenger is declared the winner only if its bootstrap CI lower bound        
    exceeds the champion's upper bound — i.e., the intervals do not overlap.
    This is a conservative criterion that guards against declaring a winner         
    on noise.                                                                       
                                                                                    
    Parameters                                                                      
    ----------  
    y_true : array-like
        Binary ground truth labels.                                                 
    prob_champ : array-like                                                         
        Predicted probabilities from the champion model.                            
    prob_chal : array-like                                                          
        Predicted probabilities from the challenger model.                          
                                                                                    
    Returns                                                                         
    -------                                                                         
    dict with keys:
        champ_auc, chal_auc : float — point estimates.
        champ_ci, chal_ci : tuple of (lower, upper) — 95% bootstrap intervals.      
        challenger_wins : bool — True if CIs are non-overlapping in challenger's favor.                                                                              
    """   
    champ_auc, champ_lower, champ_upper = bootstrap_ci(roc_auc, y_true, prob_champ)
    chal_auc, chal_lower, chal_upper = bootstrap_ci(roc_auc, y_true, prob_chal)
    return dict(
                champ_auc = champ_auc, 
                champ_ci = (champ_lower, champ_upper),
                chal_auc = chal_auc,
                chal_ci = (chal_lower, chal_upper),
                challenger_wins = chal_lower > champ_upper
            )