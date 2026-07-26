"""
HAR-RV model (Corsi 2009). Regresses future realized volatility on three lagged RV
components: daily (RV_d), weekly (RV_w = mean of last 5 days), and monthly
(RV_m = mean of last 22 days). Fits via OLS, produces point forecasts and optionally
confidence intervals.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from lib.models.realized_vol import fill_horizon_rv_metrics
from lib.utils.data_utils import get_bars

def compute_har_rv_coefficients(rvFrame: pd.DataFrame) -> dict:
    """
    Compute the coefficients for the HAR-RV model using OLS regression.
    Returns a dictionary with the coefficients for daily, weekly, and monthly RV.
    rvFrame must already contain the 'day_rv', 'week_rv', 'month_rv', and 'target' columns
    (see fill_horizon_rv_metrics).
    """
    model = sm.OLS(rvFrame['target'], sm.add_constant(rvFrame[['day_rv', 'week_rv', 'month_rv']])).fit()
    return {
        'const': model.params['const'],
        'day_rv': model.params['day_rv'],
        'week_rv': model.params['week_rv'],
        'month_rv': model.params['month_rv'],
    }


if __name__ == "__main__":
    barFrame = get_bars(symbol="SPY", start="2015-01-01", end="2019-01-01")
    print("Loaded bars shape:", barFrame.shape)
    rvFrame = fill_horizon_rv_metrics('log_rv', barFrame)
    coefficients = compute_har_rv_coefficients(rvFrame)
    print(coefficients)

    