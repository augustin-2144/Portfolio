import numpy as np
import pandas as pd

def covariance_matrix(returns):
    """ Compute covariance matrix of asset returns
    
    Parameters
    ----------
    returns : pd.DataFrame

    Returns
    -------
    cov : pd.DataFrame
    """
    return returns.cov()


def portfolio_volatility(weights, cov):
    """ Compute portfolio volatility.
    
    Parameters
    ----------
    weights : np.array
    cov : pd.DataFrame or np.array

    Returns
    -------
    float
    """
    w = np.array(weights)
    return np.sqrt(w.T @ cov @ w)


def portfolio_variance(weights, cov):
    """
    Compute portfolio variance.
    """
    w = np.array(weights)
    return w.T @ cov @ w
