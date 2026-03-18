import numpy as np
from sklearn.calibration import calibration_curve

def brier_score(y_true, y_prob):
    """
    Compute Brier Score

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Brier Score between 0 and 1.
    """
    result = np.mean((y_prob-y_true)**2)
    return result

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    Measures the average gap between predicted probabilities and
    actual positive rates across bins. A perfectly calibrated model
    has ECE of 0.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    n_bins : int, default 10
        Number of bins to use for calibration curve.

    Returns
    -------
    float
        Expected calibration error. Lower is better, 0 is perfect.
    """
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    # ECE = mean absolute difference between predicted and actual
    ece = np.mean(np.abs(mean_predicted - fraction_of_positives))
    return ece


def reliability_diagram_data(y_true, y_prob, n_bins=10):
    """
    Return calibration curve data for plotting a reliability diagram.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    n_bins : int, default 10
        Number of bins to use for calibration curve.

    Returns
    -------
    fraction_of_positives : np.ndarray
      Actual positive rate per bin.
    mean_predicted : np.ndarray
      Mean predicted probability per bin.
    """
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    return fraction_of_positives, mean_predicted
    