import numpy as np

def mae(y_true, y_pred):
    """
    Compute Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True continuous target values.
    y_pred : array-like of shape (n_samples,)
        Predicted continuous values.

    Returns
    -------
    float
        Mean absolute error. Lower is better, 0 is perfect.
    """
    return np.mean(np.abs(y_true-y_pred))


def rmse(y_true, y_pred):
    """                                                                             
    Compute Root Mean Squared Error (RMSE).                                       

    Penalizes large errors more heavily than MAE.                                   

    Parameters                                                                      
    ----------                                                                    
    y_true : array-like of shape (n_samples,)                                       
        True continuous target values.                                              
    y_pred : array-like of shape (n_samples,)                                       
        Predicted continuous values.                                                
                                                                                
    Returns                                                                         
    -------                                                                       
    float
        Root mean squared error. Lower is better, 0 is perfect.
    """ 
    return np.sqrt(np.mean((y_true-y_pred)**2))

def mape(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error (MAPE).

    Measures error as a percentage of the actual value.
    Useful when relative error matters more than absolute error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True continuous target values. Should not contain zeros.
    y_pred : array-like of shape (n_samples,)
        Predicted continuous values.

    Returns
    -------
    float
        Mean absolute percentage error. Lower is better, 0 is perfect.
        A value of 0.1 means predictions are off by 10% on average.
    """
    return np.mean(np.abs(y_true-y_pred)/np.abs(y_true))

def r2(y_true, y_pred):
    """
    Compute R-squared (coefficient of determination).

    Measures the proportion of variance in the target variable
    explained by the model. A value of 1.0 is perfect, 0.0 means
    the model performs no better than predicting the mean every time,
    and negative values mean the model is worse than the mean.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True continuous target values.
    y_pred : array-like of shape (n_samples,)
        Predicted continuous values.

    Returns
    -------
    float
        R-squared score. 1.0 is perfect, lower is worse.
    """
    SS_res = ((y_true - y_pred)**2).sum()
    SS_tot = ((y_true - np.mean(y_true))**2).sum()
    return 1 - (SS_res / SS_tot)

def theil_u(y_true, y_pred):
    """
    Compute Theil's U statistic.

    Compares model accuracy against a naive no-change forecast.
    Values less than 1 indicate the model outperforms the naive baseline.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True continuous target values.
    y_pred : array-like of shape (n_samples,)
        Predicted continuous values.

    Returns
    -------
    float
        Theil's U. < 1 means model beats naive forecast,
        1 means equal, > 1 means worse than naive.
    """
    naive_pred = y_true[:-1]
    y_true_trimmed = y_true[1:]
    rmse1 = rmse(y_true_trimmed, y_pred[1:])
    rmse2 = rmse(y_true_trimmed, naive_pred)
    return rmse1/rmse2