"""
Core backtesting event loop. Walks forward through time bar-by-bar, calls signal
logic, updates positions, and records state. Enforces no look-ahead bias: only
data up to the current timestep is exposed to downstream code at any point.
"""

from datetime import datetime, timedelta

import pandas as pd

from lib.backtester import portfolio
from lib.backtester.signals import calc_weights
from lib.models.har_rv import predict_rv, retrain_model
from lib.models.realized_vol import fill_horizon_rv_metrics
from lib.utils.data_utils import get_bars, get_rate_and_close


def run_backtest(portfolio, train_years, start, end, transaction_cost_rate=0.000945):

   
    train_start = (datetime.strptime(start, "%Y-%m-%d") - pd.DateOffset(years=train_years) - timedelta(days=45)).strftime("%Y-%m-%d")
    test_data = get_bars('SPY', train_start, end)
    metricDf = fill_horizon_rv_metrics("rv", test_data)

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    pre_start_dates = metricDf.loc[metricDf['timestamp'] < start_date, 'timestamp']
    coefficients = retrain_model(metricDf, pre_start_dates.iloc[-1], train_years, isBV=False)

    dates_rates_and_close = get_rate_and_close('SPY', start, end)

    prev_weight = 0.0
    history = []

    for index, row in dates_rates_and_close.iterrows():

        forecast = predict_rv(metricDf, row['timestamp'], coefficients, isBV=False);
        new_weight = calc_weights(forecast);


        trade_size = portfolio.calc_position_size(prev_weight, new_weight)
        portfolio.execute_trade(trade_size, row['close'], transaction_cost_rate * abs(trade_size))
        portfolio.update(row['close'], row['borrow_rate'], row['cash_rate'])

        prev_weight = new_weight
        coefficients = retrain_model(metricDf, row['timestamp'], train_years, isBV=False)

        history.append({'timestamp': row['timestamp'], 'weight': new_weight, 'forecast': forecast, **portfolio.get_state()})

    return pd.DataFrame(history)
