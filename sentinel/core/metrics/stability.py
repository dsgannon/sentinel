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