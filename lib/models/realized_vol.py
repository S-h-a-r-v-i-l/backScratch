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
        print("Warning: Realized variance is zero for this day. Check the data.")
        print(bars)
        return {
            'rv': 0,
            'log_rv': -np.inf,
            'rvol': 0,
        }
    return {
        'rv': rv,
        'log_rv': np.log(rv),
        'rvol': np.sqrt(rv),
    }

def fill_Dataframe_with_rv_metrics(bars: pd.DataFrame) -> pd.DataFrame:
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
        df.loc[len(df)] = [date, metrics['log_rv'], metrics['rv'], metrics['rvol']]
    return df


if __name__ == "__main__":
    df = get_bars(symbol="SPY", start="2024-01-01", end="2025-01-01")
    print(df['timestamp'].dtype)
    print(df.shape)
    filled_df = fill_Dataframe_with_rv_metrics(df);
    print(filled_df.head())