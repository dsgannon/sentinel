import numpy as np
from scipy.stats import false_discovery_control

def bonferroni(p_values, alpha = 0.05):
    """
    Apply Bonferroni correction for multiple testing.

    Divides the significance threshold by the number of tests,
    controlling the family-wise error rate (FWER). Conservative
    but simple — use when you want to minimize false positives.

    Parameters
    ----------
    p_values : array-like
        Array of p-values from multiple statistical tests.
    alpha : float, default 0.05
        Desired overall significance level.

    Returns
    -------
    float
        Corrected significance threshold. Reject null hypothesis
        for any test where p_value < corrected_threshold.
    """
    return alpha/ len(p_values)


def benjamini_hochberg(p_values, alpha = 0.05):
    """
    Apply Benjamini-Hochberg correction for multiple testing.

    Controls the false discovery rate (FDR) — the expected proportion
    of flagged results that are false positives. Less conservative than
    Bonferroni, better suited when many tests are run and some real
    signals are expected.

    Parameters
    ----------
    p_values : array-like
        Array of p-values from multiple statistical tests.
    alpha : float, default 0.05
        Desired false discovery rate.

    Returns
    -------
    np.ndarray
        Adjusted p-values. Reject null hypothesis where
        adjusted p-value < alpha.
    """
    return false_discovery_control(p_values)