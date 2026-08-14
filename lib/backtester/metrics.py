"""
Post-run performance analysis: Sharpe ratio, Sortino, max drawdown, hit rate,
annualized return, information ratio. Also useful for evaluating HAR-RV forecast
accuracy via QLIKE or MSE loss. All functions are pure — take a returns series,
output a number.
"""

import numpy as np
import pandas as pd

def sharpe_ratio(returns: pd.Series, risk_free_rate: float | pd.Series = 0.0) -> float:
    daily_rf = risk_free_rate if isinstance(risk_free_rate, pd.Series) else risk_free_rate / 252
    excess_returns = returns - daily_rf
    # nunique(), not std() == 0: pandas' variance algorithm produces float noise
    # (~1e-19) even on a genuinely constant series, so std can come back nonzero
    # when the true variance is exactly zero.
    if excess_returns.nunique(dropna=False) <= 1:
        return np.nan
    return (excess_returns.mean() / excess_returns.std()) * (252 ** 0.5)

def sortino_ratio(returns: pd.Series, risk_free_rate: float | pd.Series = 0.0) -> float:
    daily_rf = risk_free_rate if isinstance(risk_free_rate, pd.Series) else risk_free_rate / 252
    excess_returns = returns - daily_rf
    downside_deviation = np.sqrt((excess_returns.clip(upper=0) ** 2).mean())
    if downside_deviation == 0:
        return np.nan
    return (excess_returns.mean() / downside_deviation) * (252 ** 0.5)

def max_drawdown(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    drawdown = (running_peak - equity) / running_peak
    return drawdown.max()
