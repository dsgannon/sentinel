import numpy as np
import pandas as pd
import shap

def shap_importance(model, X):
    """                                                                             
    Compute global feature importance using mean absolute SHAP values.              
                                                                                    
    SHAP (SHapley Additive exPlanations) attributes each prediction to              
    individual features based on game-theoretic Shapley values. Global
    importance is the mean absolute SHAP value across all observations,             
    representing average feature contribution magnitude.
                                                                                    
    Parameters  
    ----------                                                                      
    model : fitted model object
        Any model supported by shap.Explainer (sklearn, XGBoost, etc.).
    X : pd.DataFrame                                                                
        Feature matrix to explain.                                                  
                                                                                    
    Returns                                                                         
    -------     
    pd.DataFrame
        Columns: feature, importance. Sorted highest to lowest.
    """ 
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    importance = np.abs(shap_values.values).mean(axis = 0)
    return pd.DataFrame({"feature": X.columns, 
                        "importance": importance
            }).sort_values("importance", 
                            ascending = False)


def shap_stability(model, snapshots):
    """         
    Track SHAP feature importances across multiple time periods.
                                                                                    
    Detects model drift by measuring whether the features driving                   
    predictions are changing over time. A feature that was important                
    at training but loses importance in production signals potential                
    concept drift.                                                                  

    Parameters                                                                      
    ----------  
    model : fitted model object
        Any model supported by shap.Explainer.
    snapshots : list of dict                                                        
        Each dict must contain:
            - "period" : str — label for the time period (e.g. "2024-Q1")           
            - "X" : pd.DataFrame — feature matrix for that period                   

    Returns                                                                         
    -------     
    pd.DataFrame
        Columns: period, feature, importance. One row per feature per period.       
    """
    rows = []
    for snapshot in snapshots:
        df = shap_importance(model, snapshot["X"])
        for _, row in df.iterrows():
            rows.append({
                "period": snapshot["period"],
                "feature": row["feature"],
                "importance": row["importance"]
            })
    return pd.DataFrame(rows)
