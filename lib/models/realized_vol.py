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
    dailyFrame = fill_daily_bipower_variation(bars)
    df = pd.DataFrame(columns=['timestamp', 'day_bv', 'week_bv', 'month_bv', 'target'])

    for i in range(len(dailyFrame)):
        if(i < 22):
            continue
        day_bv = dailyFrame.iloc[i-1][bvMetric]
        week_bv = dailyFrame.iloc[i-5:i][bvMetric].mean()
        month_bv = dailyFrame.iloc[i-22:i][bvMetric].mean()
        df.loc[len(df)] = [dailyFrame.iloc[i]['timestamp'], day_bv, week_bv, month_bv, dailyFrame.iloc[i][bvMetric]]
    return df

def fill_daily_jump(daily_rv: pd.DataFrame, daily_bv: pd.DataFrame) -> pd.DataFrame:
    merged = daily_rv.merge(daily_bv, on='timestamp', how='inner')
    jump = (merged['rv'] - merged['bv']).clip(lower=0)
    return pd.DataFrame({'timestamp': merged['timestamp'], 'jump': jump})


def fill_daily_CSP(daily_rv: pd.DataFrame, daily_jump: pd.DataFrame) -> pd.DataFrame:
    merged = daily_rv.merge(daily_jump, on='timestamp', how='inner')
    csp = merged['rv'] - merged['jump']
    return pd.DataFrame({'timestamp': merged['timestamp'], 'CSP': csp})

def fill_horizon_CSP(daily_rv: pd.DataFrame, daily_jump: pd.DataFrame) -> pd.DataFrame:
    dailyFrame = fill_daily_CSP(daily_rv, daily_jump)
    df = pd.DataFrame(columns=['timestamp', 'day_CSP', 'week_CSP', 'month_CSP', 'target'])

    for i in range(len(dailyFrame)):
        if(i < 22):
            continue
        day_CSP = dailyFrame.iloc[i-1]['CSP']
        week_CSP = dailyFrame.iloc[i-5:i]['CSP'].mean()
        month_CSP = dailyFrame.iloc[i-22:i]['CSP'].mean()
        df.loc[len(df)] = [dailyFrame.iloc[i]['timestamp'], day_CSP, week_CSP, month_CSP, dailyFrame.iloc[i]['CSP']]
    return df

def fill_horizon_jump(daily_rv: pd.DataFrame, daily_bv: pd.DataFrame) -> pd.DataFrame:
    dailyFrame = fill_daily_jump(daily_rv, daily_bv)
    df = pd.DataFrame(columns=['timestamp', 'day_Jump', 'week_Jump', 'month_Jump', 'target'])

    for i in range(len(dailyFrame)):
        if(i < 22):
            continue
        day_Jump = dailyFrame.iloc[i-1]['jump']
        week_Jump = dailyFrame.iloc[i-5:i]['jump'].mean()
        month_Jump = dailyFrame.iloc[i-22:i]['jump'].mean()
        df.loc[len(df)] = [dailyFrame.iloc[i]['timestamp'], day_Jump, week_Jump, month_Jump, dailyFrame.iloc[i]['jump']]
    return df

if __name__ == "__main__":
    df = get_bars(symbol="SPY", start="2024-01-01", end="2025-01-01")
    print(df['timestamp'].dtype)
    print(df.shape)
    filled_df = fill_horizon_rv_metrics('rv', df);
    print(filled_df.head())