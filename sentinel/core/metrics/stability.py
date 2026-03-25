import numpy as np
import pandas as pd

def performance_over_time(snapshots, metric_fn):
    """                                                                             
    Track a model metric across multiple time periods.                              
                                                                                    
    Evaluates metric_fn on each snapshot and returns a time-series                  
    DataFrame suitable for trend analysis and regulatory reporting.                 
                                                                                    
    Parameters  
    ----------                                                                      
    snapshots : list of dict                                                        
        Each dict must contain:                                                     
            - "period" : str — label for the time period (e.g. "2024-Q1")           
            - "y_true" : array-like — ground truth values                           
            - "y_pred" : array-like — model predictions or probabilities            
    metric_fn : callable                                                            
        A function with signature f(y_true, y_pred) -> float.                       
                                                                                    
    Returns                                                                         
    -------     
    pd.DataFrame
        Columns: period, metric_value. One row per snapshot.                        
    """ 
    rows = []
    for snapshot in snapshots:
        value = metric_fn(snapshot["y_true"], snapshot["y_pred"])
        rows.append({"period": snapshot["period"] , "metric_value": value})
    return pd.DataFrame(rows)

def degradation_alert(df, baseline_metric, threshold):
    """                                                                             
    Flag time periods where model performance has degraded beyond a threshold.      
                                                                                    
    Compares each period's metric value against a development-time baseline         
    and returns only the periods where degradation exceeds the threshold.           
    Intended for use with the output of performance_over_time().                    
                                                                                    
    Parameters                                                                      
    ----------                                                                      
    df : pd.DataFrame
        Output of performance_over_time(). Must contain a "metric_value" column.
    baseline_metric : float                                                         
        The metric value established at model development time.
    threshold : float                                                               
        Maximum acceptable degradation (e.g. 0.05 for a 5-point AUC drop).
                                                                                    
    Returns     
    -------                                                                         
    pd.DataFrame
        Subset of df containing only rows where degradation exceeds threshold.
        Empty DataFrame if no periods breach the threshold.
    """ 
    diff = baseline_metric - df["metric_value"]
    return df[diff > threshold]

def vintage_analysis(df, cohort_col, period_col, y_true_col, y_pred_col, metric_fn):
    """
    Track model performance by origination cohort across reporting periods.

    Standard in insurance and lending validation — groups observations by
    when they entered the portfolio (cohort) and tracks how each cohort's
    model performance evolves over time. Reveals whether the model degrades
    faster for older cohorts as economic conditions change.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with all observations, cohort labels, and predictions.
    cohort_col : str
        Column name identifying the origination cohort (e.g. "origination_quarter").
    period_col : str
        Column name identifying the reporting period (e.g. "reporting_quarter").
    y_true_col : str
        Column name for ground truth labels.
    y_pred_col : str
        Column name for model predictions or probabilities.
    metric_fn : callable
        A function with signature f(y_true, y_pred) -> float.

    Returns
    -------
    pd.DataFrame
        Columns: cohort, period, metric_value.
        One row per cohort-period combination with at least 10 observations.
    """
    rows = []
    for cohort in df[cohort_col].unique():
        for period in df[period_col].unique():
            mask = (df[cohort_col] == cohort) & (df[period_col] == period)
            if mask.sum() < 10:
                continue
            score = metric_fn(df[y_true_col][mask], df[y_pred_col][mask])
            rows.append({"cohort": cohort, "period": period, "metric_value": score})
    return pd.DataFrame(rows)