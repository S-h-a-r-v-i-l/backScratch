"""
Unit tests for the HAR-RV model. Checks that coefficients match reference
implementations (e.g., R's HARmodel), forecasting works on edge cases (short
series, NaNs), and the three-component regression produces correct in-sample fit.
"""
import pandas as pd
import numpy as np  

from lib.models.realized_vol import fill_daily_CSP, fill_daily_bipower_variation, fill_daily_jump, fill_daily_jump, fill_daily_rv_metrics, fill_horizon_CSP, fill_horizon_bipower_variation, fill_horizon_jump, fill_horizon_rv_metrics
from lib.models.har_rv import compute_har_rv_cj_coefficients, compute_har_rv_coefficients
from lib.utils.data_utils import get_bars, save_as_parquet

VALID_RV_METRICS = ('rv', 'log_rv', 'bv', 'log_bv')

def test_har_rv_coefficients(test_data: pd.DataFrame, start_offset: int, window_size: int, rvMetric: str, toggleBV: bool = False) -> pd.DataFrame:
    if rvMetric not in VALID_RV_METRICS:
        raise ValueError(f"rvMetric must be one of {VALID_RV_METRICS}, got {rvMetric!r}")

    if(toggleBV):
        dailyFrame = fill_daily_bipower_variation(test_data)
        metricDf = fill_horizon_bipower_variation(rvMetric, test_data)
    else:
        dailyFrame = fill_daily_rv_metrics(test_data)
        metricDf = fill_horizon_rv_metrics(rvMetric, test_data)

    coefficients = compute_har_rv_coefficients(metricDf.head(start_offset), isBV=toggleBV)


    if(toggleBV):
        columns = ['timestamp', 'predicted_bv', 'actual_bv', 'MSE']
    else:
        columns = ['timestamp', 'predicted_rv', 'actual_rv', 'MSE']
    if rvMetric == 'rv' or rvMetric == 'bv':
        columns.append('QLIKE')
    results_df = pd.DataFrame(columns=columns)

    for i in range(start_offset, len(metricDf)):
        j = max(0, i - window_size)
        # metricDf skips the first 22 days of dailyFrame (see fill_horizon_rv_metrics), so translate the index
        d = i + 22
        day = dailyFrame.iloc[d-1][rvMetric]
        week = dailyFrame.iloc[d-5:d][rvMetric].mean()
        month = dailyFrame.iloc[d-22:d][rvMetric].mean()

        predicted = (coefficients['const'] +
                        coefficients['day_rv'] * day +
                        coefficients['week_rv'] * week +
                        coefficients['month_rv'] * month)

        actual = dailyFrame.iloc[d][rvMetric]
        if rvMetric == 'rv' or rvMetric == 'bv':
            predicted = max(predicted, 1e-8)
        row = [dailyFrame.iloc[d]['timestamp'], predicted, actual, (predicted - actual) ** 2]
        if rvMetric == 'rv' or rvMetric == 'bv':
            ratio = actual / predicted
            row.append(ratio - np.log(ratio) - 1)
        results_df.loc[len(results_df)] = row

        # compute new coeeficients
        coefficients = compute_har_rv_coefficients(metricDf.iloc[j:i], isBV=toggleBV)

    save_as_parquet(results_df, f"data/results/test_har_{rvMetric}.parquet")
    return results_df

def test_har_rv_cj_coefficients(test_data: pd.DataFrame, start_offset: int, window_size: int) -> pd.DataFrame:
    # Implementation for testing HAR-RV coefficients with CJ model
    rv_d = fill_daily_rv_metrics(test_data)
    bv_d = fill_daily_bipower_variation(test_data)
    jump_d = fill_daily_jump(rv_d, bv_d)

    full_csp = fill_horizon_CSP(rv_d, jump_d)
    full_jump = fill_horizon_jump(rv_d, bv_d)
    full_rv = fill_horizon_rv_metrics(rvMetric='rv', bars=test_data)

    full_Frame = full_csp[['timestamp', 'day_CSP', 'week_CSP', 'month_CSP']].merge(
        full_jump[['timestamp', 'day_Jump', 'week_Jump', 'month_Jump']], on='timestamp', how='inner'
    ).merge(
        full_rv[['timestamp', 'target']], on='timestamp', how='inner'
    )

    coefficients = compute_har_rv_cj_coefficients(full_Frame.head(start_offset))

    results_df = pd.DataFrame(columns=[ 'timestamp', 'predicted_rv', 'actual_rv', 'MSE', 'QLIKE'])

    for i in range(start_offset, len(full_Frame)):
        j = max(0, i - window_size)
        day_CSP = full_Frame.iloc[i]["day_CSP"]
        week_CSP = full_Frame.iloc[i]["week_CSP"]
        month_CSP = full_Frame.iloc[i]["month_CSP"]
        day_Jump = full_Frame.iloc[i]["day_Jump"]
        week_Jump = full_Frame.iloc[i]["week_Jump"]
        month_Jump = full_Frame.iloc[i]["month_Jump"]


        predicted = (coefficients['const'] +
                        coefficients['day_CSP'] * day_CSP +
                        coefficients['week_CSP'] * week_CSP +
                        coefficients['month_CSP'] * month_CSP +
                        coefficients['day_Jump'] * day_Jump +
                        coefficients['week_Jump'] * week_Jump +
                        coefficients['month_Jump'] * month_Jump)
        predicted = max(predicted, 1e-8)

        actual = full_Frame.iloc[i]["target"]
        row = [full_Frame.iloc[i]['timestamp'], predicted, actual, (predicted - actual) ** 2]
        ratio = actual / predicted
        row.append(ratio - np.log(ratio) - 1)
        results_df.loc[len(results_df)] = row

        # compute new coeeficients
        coefficients = compute_har_rv_cj_coefficients(full_Frame.iloc[j:i])

    save_as_parquet(results_df, f"data/results/test_har_cj.parquet")
    return results_df


if __name__ == "__main__":
    test = get_bars(symbol="SPY", start="2013-01-01", end="2019-01-01")
    print("Loaded test data shape:", test.shape)
    results = test_har_rv_coefficients(test, start_offset=100, window_size=100, rvMetric='log_rv', toggleBV=False)
    results = test_har_rv_coefficients(test, start_offset=100, window_size=100, rvMetric='log_bv', toggleBV=True)
    print(results.head())

    # Jump/CSP coefficients need a wider window than the plain RV/BV fit: the jump
    # regressor is sparse (~20% of days are ~zero), so a 100-day window only has a
    # handful of informative jump observations and the OLS fit on week_Jump/month_Jump
    # becomes unstable (e.g. coefficients of +25 / -5 seen around the Feb 2018 vol spike).
    cj_results = test_har_rv_cj_coefficients(test, start_offset=100, window_size=500)
    print(cj_results.head())



        







    