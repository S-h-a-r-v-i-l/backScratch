"""
Core backtesting event loop. Walks forward through time bar-by-bar, calls signal
logic, updates positions, and records state. Enforces no look-ahead bias: only
data up to the current timestep is exposed to downstream code at any point.
"""

from datetime import datetime, timedelta

import pandas as pd

from lib.backtester.signals import calc_weights
from lib.models.har_rv import predict_rv, retrain_model
from lib.models.realized_vol import fill_horizon_rv_metrics
from lib.utils.data_utils import get_bars, get_rate_and_close, save_as_parquet


def run_backtest(portfolio, train_years, start, end, transaction_cost_rate=0.0000945,
                 rebalance_band=0.3, results_path=None, rate_vintage_date=None):

   
    train_start = (datetime.strptime(start, "%Y-%m-%d") - pd.DateOffset(years=train_years) - timedelta(days=45)).strftime("%Y-%m-%d")
    test_data = get_bars('SPY', train_start, end)
    metricDf = fill_horizon_rv_metrics("rv", test_data)

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    pre_start_dates = metricDf.loc[metricDf['timestamp'] < start_date, 'timestamp']
    coefficients = retrain_model(metricDf, pre_start_dates.iloc[-1], train_years, isBV=False)

    dates_rates_and_close = get_rate_and_close('SPY', start, end, rate_vintage_date=rate_vintage_date)

    valid_dates = set(metricDf['timestamp'])
    dates_rates_and_close = dates_rates_and_close[dates_rates_and_close['timestamp'].dt.date.isin(valid_dates)]

    prev_weight = 0.0
    risk_free_equity = portfolio.equity
    history = []
    last_refit_week = None

    for index, row in dates_rates_and_close.iterrows():

        current_day = row['timestamp'].date()
        visible_metrics = metricDf.loc[metricDf['timestamp'] <= current_day]

        forecast = predict_rv(visible_metrics, row['timestamp'], coefficients, isBV=False);
        target_weight = calc_weights(forecast);

        # Rebalance band: only move the book when the target weight has drifted
        # at least `rebalance_band` from what we're currently holding. Day-to-day
        # wiggle in the RV forecast is mostly sampling noise, and rebalancing to
        # it every day just pays transaction costs to chase that noise.
        if abs(target_weight - prev_weight) >= rebalance_band:
            new_weight = target_weight
        else:
            new_weight = prev_weight

        trade_size = portfolio.calc_position_size(prev_weight, new_weight)
        portfolio.execute_trade(trade_size, row['close'], transaction_cost_rate * abs(trade_size))
        portfolio.update(row['close'], row['borrow_rate'], row['cash_rate'])

        prev_weight = new_weight

        # Weekly re-estimation: the HAR coefficients move negligibly from one day
        # to the next, so a daily OLS refit mostly fits sampling noise. Refit on
        # the first trading day of each ISO week instead.
        iso = current_day.isocalendar()
        current_week = (iso[0], iso[1])
        if current_week != last_refit_week:
            coefficients = retrain_model(visible_metrics, current_day, train_years, isBV=False)
            last_refit_week = current_week

        risk_free_equity *= (1 + row['risk_free_rate'])

        history.append({
            'timestamp': row['timestamp'],
            'weight': new_weight,
            'forecast': forecast,
            **portfolio.get_state(),
            'risk_free_rate': row['risk_free_rate'],
            'risk_free_equity': risk_free_equity,
        })

    history_df = pd.DataFrame(history)

    if results_path is not None:
        save_as_parquet(history_df, results_path)

    return history_df
