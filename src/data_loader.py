import yfinance as yf
import pandas as pd

def load_prices(tickers, start, end):
    """ Download adjusted close prices from Yahoo Finance.
    Parameters
    ----------
    tickers : list of str
    start : str (YYYY-MM-DD)
    end : str (YYYY-MM-DD)

    Returns
    -------
    prices : pd.DataFrame
    """
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    return data

def compute_returns(prices):
    """ Compute daily returns from price data"""
    returns = prices.pct_change(fill_method=None).dropna()
    return returns
