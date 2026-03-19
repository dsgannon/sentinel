import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def disparate_impact_ratio(y_pred, sensitive_attr, privileged_value):
     """                                                                             
    Compute the disparate impact ratio between unprivileged and privileged groups.  
                                                                                    
    The "80% rule": a ratio below 0.8 indicates potential discriminatory impact     
    against the unprivileged group, as defined under ECOA and Fair Housing Act.
                                                                                    
    Parameters  
    ----------                                                                      
    y_pred : array-like
        Binary model predictions (0/1).
    sensitive_attr : array-like                                                     
        Group label for each observation (e.g. "white", "black").
    privileged_value : scalar                                                       
        The reference group label (e.g. "white").                                   
                                                                                    
    Returns                                                                         
    -------                                                                         
    float                                                                           
        Ratio of unprivileged positive rate to privileged positive rate.
        Values below 0.8 warrant regulatory scrutiny.                               
    """ 
    y_pred = np.array(y_pred)
    sensitive_attr = np.array(sensitive_attr)                                           
    pos_rate_unpriv = np.mean(y_pred[sensitive_attr != privileged_value])
    pos_rate_priv = np.mean(y_pred[sensitive_attr == privileged_value])
    return pos_rate_unpriv / pos_rate_priv

def demographic_parity_difference(y_pred, sensitive_attr, privileged_value):
    """                                                                             
    Compute the demographic parity difference between unprivileged and privileged groups.                                                                             
                
    Additive counterpart to disparate_impact_ratio. A value of 0 indicates          
    perfect demographic parity. Negative values indicate the unprivileged
    group receives fewer positive predictions.                                      
                                                                                    
    Parameters                                                                      
    ----------                                                                      
    y_pred : array-like
        Binary model predictions (0/1).
    sensitive_attr : array-like
        Group label for each observation.                                           
    privileged_value : scalar                                                       
        The reference group label.                                                  
                                                                                    
    Returns     
    -------                                                                         
    float       
        Difference in positive prediction rates (unprivileged minus privileged).
    """ 
    y_pred = np.array(y_pred)
    sensitive_attr = np.array(sensitive_attr)                                           
    pos_rate_unpriv = np.mean(y_pred[sensitive_attr != privileged_value])
    pos_rate_priv = np.mean(y_pred[sensitive_attr == privileged_value])
    return pos_rate_unpriv - pos_rate_priv

def equalized_odds(y_true, y_pred, sensitive_attr, privileged_value):
    """                                                                             
    Measure equalized odds across privileged and unprivileged groups.
                                                                                    
    A model satisfies equalized odds if TPR and FPR are equal across groups.        
    A model can have equal positive prediction rates but still systematically       
    make different error types for different groups — this metric detects that.     
                                                                                    
    Parameters                                                                      
    ----------                                                                      
    y_true : array-like
        Binary ground truth labels.                                                 
    y_pred : array-like                                                             
        Binary model predictions (0/1).                                             
    sensitive_attr : array-like                                                     
        Group label for each observation.
    privileged_value : scalar
        The reference group label.                                                  
                                                                                    
    Returns                                                                         
    -------                                                                         
    dict with keys:
        priv_tpr, priv_fpr : float — TPR and FPR for the privileged group.
        unpriv_tpr, unpriv_fpr : float — TPR and FPR for the unprivileged group.    
        equalized_odds_gap : float — average of absolute TPR and FPR differences.
            Values near 0 indicate fairness; larger values warrant investigation.   
    """ 
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)                                                           
    sensitive_attr = np.array(sensitive_attr)
    mask_priv = sensitive_attr == privileged_value
    cm_priv = confusion_matrix(y_true[mask_priv], y_pred[mask_priv])      
    mask_unpriv = sensitive_attr != privileged_value
    cm_unpriv = confusion_matrix(y_true[mask_unpriv], y_pred[mask_unpriv])               
    priv_tpr = cm_priv[1,1] / (cm_priv[1,1] + cm_priv[1,0])
    priv_fpr = cm_priv[0,1] / (cm_priv[0,1] + cm_priv[0,0]) 
    unpriv_tpr = cm_unpriv[1,1] / (cm_unpriv[1,1] + cm_unpriv[1,0])
    unpriv_fpr = cm_unpriv[0,1] / (cm_unpriv[0,1] + cm_unpriv[0,0]) 
    equalized_odds_gap = np.mean([np.abs(unpriv_tpr - priv_tpr), np.abs(unpriv_fpr - priv_fpr)])
    return dict(
                priv_tpr = priv_tpr, 
                priv_fpr = priv_fpr, 
                unpriv_tpr = unpriv_tpr, 
                unpriv_fpr = unpriv_fpr, 
                equalized_odds_gap = equalized_odds_gap
            )
