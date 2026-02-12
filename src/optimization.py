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
