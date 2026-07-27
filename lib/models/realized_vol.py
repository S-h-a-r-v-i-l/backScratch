"""
RV estimators: compute realized volatility from raw intraday price data.
Includes standard RV (sum of squared intraday returns), bipower variation
(jump-robust), and kernel-based estimators. Kept separate from har_rv.py so
estimators can be swapped without touching the model and tested independently.
"""
import pandas as pd
import numpy as np

from lib.utils.data_utils import get_bars

def compute_daily_rv_metrics(bars: pd.DataFrame) -> dict:
    log_returns = np.log(bars['close'] / bars['close'].shift(1)).dropna()
    rv = (log_returns ** 2).sum()
    if(rv == 0):
        return None
    return {
        'rv': rv,
        'log_rv': np.log(rv),
        'rvol': np.sqrt(rv),
    }

def compute_daily_bipower_variation(bars: pd.DataFrame) -> dict:
    log_returns = np.log(bars['close'] / bars['close'].shift(1)).dropna()
    abs_log_returns = np.abs(log_returns)
    bv = (np.pi / 2) * (abs_log_returns * abs_log_returns.shift(1)).dropna().sum()
    if(bv == 0):
        return None
    return {
        'bv': bv,
        'log_bv': np.log(bv),
        'bvol': np.sqrt(bv),
    }

def fill_daily_rv_metrics(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Fill the DataFrame with realized volatility metrics computed from intraday data.
    """
    df = pd.DataFrame(columns=['timestamp', 'log_rv', 'rv', 'rvol'])

    grouped = bars.groupby(bars['timestamp'].dt.date)
    print(len(grouped))
    for date, group in grouped:
        if(len(group) < 2):
            print(f"Warning: Not enough data for {date}. Skipping.")
            continue
        metrics = compute_daily_rv_metrics(group)
        if metrics is None:
            print(f"Warning: Realized variance is zero for {date}. Skipping.")
            continue
        df.loc[len(df)] = [date, metrics['log_rv'], metrics['rv'], metrics['rvol']]
    return df

def fill_daily_bipower_variation(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Fill the DataFrame with bipower variation metrics computed from intraday data.
    """
    df = pd.DataFrame(columns=['timestamp', 'log_bv', 'bv', 'bvol'])

    grouped = bars.groupby(bars['timestamp'].dt.date)
    print(len(grouped))
    for date, group in grouped:
        if(len(group) < 2):
            print(f"Warning: Not enough data for {date}. Skipping.")
            continue
        metrics = compute_daily_bipower_variation(group)
        if metrics is None:
            print(f"Warning: Bipower variation is zero for {date}. Skipping.")
            continue
        df.loc[len(df)] = [date, metrics['log_bv'], metrics['bv'], metrics['bvol']]
    return df

def fill_horizon_rv_metrics(rvMetric: str, bars: pd.DataFrame) -> pd.DataFrame:
    """
    Fill the DataFrame with realized volatility metrics for all days in the input DataFrame.
    """
    dailyFrame = fill_daily_rv_metrics(bars);
    df = pd.DataFrame(columns=['timestamp', 'day_rv', 'week_rv', 'month_rv', 'target'])

    for i in range(len(dailyFrame)):
        if(i < 22):
            continue
        day_rv = dailyFrame.iloc[i-1][rvMetric]
        week_rv = dailyFrame.iloc[i-5:i][rvMetric].mean()
        month_rv = dailyFrame.iloc[i-22:i][rvMetric].mean()
        df.loc[len(df)] = [dailyFrame.iloc[i]['timestamp'], day_rv, week_rv, month_rv, dailyFrame.iloc[i][rvMetric]]
    return df

def fill_horizon_bipower_variation(bvMetric: str, bars: pd.DataFrame) -> pd.DataFrame:
    """
    Fill the DataFrame with bipower variation metrics for all days in the input DataFrame.
    """
    dailyFrame = fill_daily_bipower_variation(bars);
    df = pd.DataFrame(columns=['timestamp', 'day_bv', 'week_bv', 'month_bv', 'target'])

    for i in range(len(dailyFrame)):
        if(i < 22):
            continue
        day_bv = dailyFrame.iloc[i-1][bvMetric]
        week_bv = dailyFrame.iloc[i-5:i][bvMetric].mean()
        month_bv = dailyFrame.iloc[i-22:i][bvMetric].mean()
        df.loc[len(df)] = [dailyFrame.iloc[i]['timestamp'], day_bv, week_bv, month_bv, dailyFrame.iloc[i][bvMetric]]
    return df


if __name__ == "__main__":
    df = get_bars(symbol="SPY", start="2024-01-01", end="2025-01-01")
    print(df['timestamp'].dtype)
    print(df.shape)
    filled_df = fill_horizon_rv_metrics('rv', df);
    print(filled_df.head())