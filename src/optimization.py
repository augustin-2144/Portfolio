import numpy as np
from scipy.optimize import minimize

def portfolio_return(weights, expected_returns):
    """ Compute portfolio expected return
    
    Parameters
    ----------
    weights : np.array
    expected_returns : np.array or pd.Series
    """
    return np.dot(weights, expected_returns)


def portfolio_volatility(weights, cov_matrix):
    """ Compute portfolio volatility """
    return np.sqrt(weights.T @ cov_matrix @ weights)


def minimize_variance(expected_returns, cov_matrix):
    """ Compute minimum variance portfolio """
    n = len(expected_returns)
    init_guess = np.ones(n) / n

    bounds = tuple((0, 1) for _ in range(n))
    constraints = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    })

    result = minimize(
        portfolio_volatility,
        init_guess,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result.x


def efficient_portfolio(target_return, expected_returns, cov_matrix):
    """ Compute portfolio with minimum variance for a given target return """
    n = len(expected_returns)
    init_guess = np.ones(n) / n

    bounds = tuple((0, 1) for _ in range(n))

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: portfolio_return(w, expected_returns) - target_return}
    )

    result = minimize(
        portfolio_volatility,
        init_guess,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result.x


def sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.0):
    """ Compute portfolio Sharpe ratio """
    port_ret = portfolio_return(weights, expected_returns)
    port_vol = portfolio_volatility(weights, cov_matrix)
    return (port_ret - risk_free_rate) / port_vol


def maximize_sharpe(expected_returns, cov_matrix, risk_free_rate=0.0):
    """ Compute portfolio that maximizes Sharpe ratio """
    n = len(expected_returns)
    init_guess = np.ones(n) / n

    bounds = tuple((0, 1) for _ in range(n))
    constraints = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    })

    def negative_sharpe(w):
        return -sharpe_ratio(w, expected_returns, cov_matrix, risk_free_rate)

    result = minimize(
        negative_sharpe,
        init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result.x
