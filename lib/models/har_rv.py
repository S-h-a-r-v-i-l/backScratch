"""
HAR-RV model (Corsi 2009). Regresses future realized volatility on three lagged RV
components: daily (RV_d), weekly (RV_w = mean of last 5 days), and monthly
(RV_m = mean of last 22 days). Fits via OLS, produces point forecasts and optionally
confidence intervals.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from lib.models.realized_vol import (
    fill_horizon_rv_metrics,
    fill_daily_rv_metrics,
    fill_daily_bipower_variation,
    fill_daily_jump,
    fill_horizon_CSP,
    fill_horizon_Jump,
)
from lib.utils.data_utils import get_bars

def compute_har_rv_coefficients(rvFrame: pd.DataFrame, isBV: bool) -> dict:
    if(isBV):
        day = 'day_bv'
        week = 'week_bv'
        month = 'month_bv'
    else:
        day = 'day_rv'
        week = 'week_rv'
        month = 'month_rv'

    model = sm.OLS(rvFrame['target'], sm.add_constant(rvFrame[[day, week, month]])).fit()
    return {
        'const': model.params['const'],
        'day_rv': model.params[day],
        'week_rv': model.params[week],
        'month_rv': model.params[month],
    }

def compute_har_rv_cj_coefficients(CSP_Frame: pd.DataFrame, jump_Frame: pd.DataFrame, rv_Frame: pd.DataFrame) -> dict:
    fullFrame = CSP_Frame[['timestamp', 'day_CSP', 'week_CSP', 'month_CSP']].merge(
        jump_Frame[['timestamp', 'day_Jump', 'week_Jump', 'month_Jump']], on='timestamp', how='inner'
    ).merge(
        rv_Frame[['timestamp', 'target']], on='timestamp', how='inner'
    )

    model = sm.OLS(fullFrame['target'], sm.add_constant(fullFrame[['day_CSP', 'week_CSP', 'month_CSP', 'day_Jump', 'week_Jump', 'month_Jump']])).fit()
    return {
        'const': model.params['const'],
        'day_CSP': model.params['day_CSP'],
        'week_CSP': model.params['week_CSP'],
        'month_CSP': model.params['month_CSP'],
        'day_Jump': model.params['day_Jump'],
        'week_Jump': model.params['week_Jump'],
        'month_Jump': model.params['month_Jump']
    }



if __name__ == "__main__":
    barFrame = get_bars(symbol="SPY", start="2015-01-01", end="2019-01-01")
    print("Loaded bars shape:", barFrame.shape)
    rvFrame = fill_horizon_rv_metrics('log_rv', barFrame)
    coefficients = compute_har_rv_coefficients(rvFrame, isBV=False)
    print(coefficients)

    dailyRvFrame = fill_daily_rv_metrics(barFrame)
    dailyBvFrame = fill_daily_bipower_variation(barFrame)
    dailyJumpFrame = fill_daily_jump(dailyRvFrame, dailyBvFrame)

    cspFrame = fill_horizon_CSP(dailyRvFrame, dailyJumpFrame)
    jumpFrame = fill_horizon_Jump(dailyRvFrame, dailyBvFrame)
    rawRvFrame = fill_horizon_rv_metrics('rv', barFrame)

    cjCoefficients = compute_har_rv_cj_coefficients(cspFrame, jumpFrame, rawRvFrame)
    print(cjCoefficients)

    