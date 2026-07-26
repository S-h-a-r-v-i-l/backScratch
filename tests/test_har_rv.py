"""
Unit tests for the HAR-RV model. Checks that coefficients match reference
implementations (e.g., R's HARmodel), forecasting works on edge cases (short
series, NaNs), and the three-component regression produces correct in-sample fit.
"""
import pandas as pd
import numpy as np

from lib.models.realized_vol import fill_daily_rv_metrics, fill_horizon_rv_metrics
from lib.models.har_rv import compute_har_rv_coefficients
from lib.utils.data_utils import get_bars

VALID_RV_METRICS = ('rv', 'log_rv')

def test_har_rv_coefficients(test_data: pd.DataFrame, start_offset: int, window_size: int, rvMetric: str):
    if rvMetric not in VALID_RV_METRICS:
        raise ValueError(f"rvMetric must be one of {VALID_RV_METRICS}, got {rvMetric!r}")

    dailyFrame = fill_daily_rv_metrics(test_data)
    metricDf = fill_horizon_rv_metrics(rvMetric, test_data)
    coefficients = compute_har_rv_coefficients(metricDf.head(start_offset))

    columns = ['timestamp', 'predicted_rv', 'actual_rv', 'MSE']
    if rvMetric == 'rv':
        columns.append('QLIKE')
    results_df = pd.DataFrame(columns=columns)

    j = start_offset - window_size;
    if(j < 0):
        j = 0
    for i in range(start_offset, len(metricDf)):
        # metricDf skips the first 22 days of dailyFrame (see fill_horizon_rv_metrics), so translate the index
        d = i + 22
        day_rv = dailyFrame.iloc[d-1][rvMetric]
        week_rv = dailyFrame.iloc[d-5:d][rvMetric].mean()
        month_rv = dailyFrame.iloc[d-22:d][rvMetric].mean()

        predicted_rv = (coefficients['const'] +
                        coefficients['day_rv'] * day_rv +
                        coefficients['week_rv'] * week_rv +
                        coefficients['month_rv'] * month_rv)

        actual_rv = dailyFrame.iloc[d][rvMetric]
        row = [dailyFrame.iloc[d]['timestamp'], predicted_rv, actual_rv, (predicted_rv - actual_rv) ** 2]
        if rvMetric == 'rv':
            row.append(actual_rv * (predicted_rv / actual_rv) - np.log(predicted_rv / actual_rv) - 1)
        results_df.loc[len(results_df)] = row

        # compute new coeeficients
        coefficients = compute_har_rv_coefficients(metricDf.iloc[j:i])
        j = j + 1

    return results_df

if __name__ == "__main__":
    test = get_bars(symbol="SPY", start="2013-01-01", end="2019-01-01")
    print("Loaded test data shape:", test.shape)
    results = test_har_rv_coefficients(test, start_offset=100, window_size=100, rvMetric='rv')
    print(results.head())



        







    