"""
Tests the backtesting engine's integrity. Verifies no look-ahead bias (engine at
time T cannot access prices at T+1), P&L accumulates correctly, transaction costs
apply properly, and the event loop handles edge cases (gaps, early termination).
"""

import numpy as np
import pandas as pd
import pytest

# --- Fixture: deterministic, monkeypatch-ready replacements for get_bars and
# get_rate_and_close -----------------------------------------------------------
#
# Both fake functions derive prices from the same underlying path (`daily_close`),
# indexed off a fixed epoch rather than off whatever `start`/`end` a given call
# happens to request. That matters because engine.py calls get_bars with a
# `train_start` that shifts depending on `train_years`, and calls
# get_rate_and_close with the (separate) `start`/`end` test window — if the price
# path weren't anchored to the calendar date itself, the two calls could disagree
# about what a given date's price was.
#
# Note: engine.py never actually cross-references the two — trades execute at
# get_rate_and_close's `close`, while get_bars' intraday closes only ever feed
# HAR-RV's realized-variance features. They don't need to match each other, just
# each be internally consistent and deterministic.
#
# To use: monkeypatch.setattr("lib.backtester.engine.get_bars", make_fake_bars)
# and monkeypatch.setattr("lib.backtester.engine.get_rate_and_close", make_fake_rates)
# — patch the names as imported into engine.py, not the originals in data_utils.py,
# since `from ... import get_bars` binds a local reference in engine's namespace.

_EPOCH = pd.Timestamp("2000-01-01").date()


def _trading_day_index(date) -> int:
    return int(np.busday_count(_EPOCH, pd.Timestamp(date).date()))


def daily_close(date, base_price: float = 100.0, daily_drift: float = 0.0005) -> float:
    """The one price path both fake data sources are derived from."""
    return base_price * (1 + daily_drift) ** _trading_day_index(date)


def make_fake_bars(symbol: str, start: str, end: str, timeframe_minutes: int = 5,
                    bars_per_day: int = 3, wiggle: float = 0.001) -> pd.DataFrame:
    """
    Drop-in replacement for data_utils.get_bars. Generates deterministic intraday
    bars for every business day in [start, end]. The intraday wiggle direction
    alternates day-to-day (even/odd trading-day index) so day_rv/week_rv/month_rv
    aren't perfectly collinear — a fixture with identical RV every day would make
    the HAR-RV OLS fit singular.
    """
    dates = pd.bdate_range(start, end)
    rows = []
    for date in dates:
        close = daily_close(date)
        step = wiggle if _trading_day_index(date) % 2 == 0 else -wiggle
        for i in range(bars_per_day):
            price = close * (1 + step * (i - bars_per_day // 2))
            rows.append({
                'timestamp': pd.Timestamp(date) + pd.Timedelta(minutes=timeframe_minutes * i),
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 1000,
            })
    return pd.DataFrame(rows)


def make_fake_rates(symbol: str, start: str, end: str, base_rate: float = 1.5) -> pd.DataFrame:
    """
    Drop-in replacement for data_utils.get_rate_and_close. Same price path as
    make_fake_bars, plus constant cash/borrow/risk-free rates computed with the
    real formula (base_rate +/- 1.5% spread) so downstream rate math is exercised
    the same way it would be against real FRED data.
    """
    dates = pd.bdate_range(start, end)
    df = pd.DataFrame({
        'timestamp': dates,
        'close': [daily_close(d) for d in dates],
    })
    df['cash_rate'] = (1 + (base_rate - 1.5) / 100) ** (1 / 252) - 1
    df['borrow_rate'] = (1 + (base_rate + 1.5) / 100) ** (1 / 252) - 1
    df['risk_free_rate'] = (1 + base_rate / 100) ** (1 / 252) - 1
    return df


_SPLIT_DATE = pd.Timestamp("2009-12-31").date()
_SHOCKED_WIGGLE = 0.05


def make_altered_bars(symbol: str, start: str, end: str, timeframe_minutes: int = 5) -> pd.DataFrame:
    """Same as make_fake_bars, but days after _SPLIT_DATE get a much larger intraday
    wiggle instead of the base fixture's wiggle. Used to prove the engine can't see
    "future" bars when computing a given day's forecast/trade — history before
    _SPLIT_DATE is untouched, everything after it is deliberately made wildly
    different from the base fixture.

    Note: a uniform price *multiplier* (e.g. scaling every OHLC value after the
    split by a constant factor) does NOT work here — realized variance is computed
    from intraday log returns, which are scale-invariant to multiplying a whole
    day's prices by a constant. The perturbation has to change intraday
    variability (wiggle) itself, or the RV features — and everything downstream of
    them — end up numerically identical regardless of the "shock"."""
    base = make_fake_bars(symbol, start, end, timeframe_minutes)
    shocked = make_fake_bars(symbol, start, end, timeframe_minutes, wiggle=_SHOCKED_WIGGLE)
    future = base['timestamp'].dt.date > _SPLIT_DATE
    return pd.concat([base[~future], shocked[future]], ignore_index=True)


def test_no_lookahead_bias(monkeypatch):
    """
    Runs the same backtest twice — once against the base fixture, once against a
    version where everything after _SPLIT_DATE is shocked to a wildly different
    price. If the engine has no look-ahead bias, every row dated on/before
    _SPLIT_DATE must come out identical between the two runs (nothing after that
    date could have influenced it), while rows after _SPLIT_DATE must differ
    (otherwise the shock never actually reached the model, and the first
    assertion would be vacuously true).
    """
    from lib.backtester import engine
    from lib.backtester.portfolio import Portfolio

    def run(bars_fn):
        monkeypatch.setattr(engine, "get_bars", bars_fn)
        monkeypatch.setattr(engine, "get_rate_and_close", make_fake_rates)
        portfolio = Portfolio(cash=100_000, shares=0, equity=100_000, peak_equity=100_000, cumulative_pnl=0)
        return engine.run_backtest(portfolio, train_years=1, start="2005-01-01", end="2010-06-01")

    base_result = run(make_fake_bars)
    altered_result = run(make_altered_bars)

    split = pd.Timestamp(_SPLIT_DATE)

    base_before = base_result.loc[base_result['timestamp'] <= split].reset_index(drop=True)
    altered_before = altered_result.loc[altered_result['timestamp'] <= split].reset_index(drop=True)
    pd.testing.assert_frame_equal(base_before, altered_before, check_exact=True)

    base_after = base_result.loc[base_result['timestamp'] > split].reset_index(drop=True)
    altered_after = altered_result.loc[altered_result['timestamp'] > split].reset_index(drop=True)
    assert not base_after.equals(altered_after)


def test_pnl_accuracy(monkeypatch):
    """
    Verifies the engine's trade-sizing and mark-to-market math against an
    independently hand-computed equity trajectory. predict_rv and calc_weights
    are both faked so the weight the engine trades to each day is a known,
    fixed value instead of falling out of the real HAR-RV regression and the
    env-var-dependent vol-targeting formula — this test is about Portfolio
    bookkeeping, not forecast correctness.

    transaction_cost_rate=0 and the fixture's default base_rate=1.5 (which
    makes cash_rate exactly 0, since the +/-1.5% spread cancels) mean the only
    things that can move equity are trades and price changes — interest
    accrual and transaction costs are covered by separate tests.
    """
    from lib.backtester import engine
    from lib.backtester.portfolio import Portfolio

    weights = [0.5, 0.5, -0.3, 0.2]

    weight_iter = iter(weights)
    monkeypatch.setattr(engine, "predict_rv", lambda *args, **kwargs: next(weight_iter))
    monkeypatch.setattr(engine, "calc_weights", lambda forecast: forecast)
    monkeypatch.setattr(engine, "get_bars", make_fake_bars)
    monkeypatch.setattr(engine, "get_rate_and_close", make_fake_rates)

    portfolio = Portfolio(cash=100_000.0, shares=0.0, equity=100_000.0, peak_equity=100_000.0, cumulative_pnl=0.0)
    result = engine.run_backtest(portfolio, train_years=1, start="2020-06-01", end="2020-06-04",
                                  transaction_cost_rate=0.0)

    dates = list(pd.bdate_range("2020-06-01", "2020-06-04").date)
    prices = [daily_close(d) for d in dates]
    assert len(prices) == len(weights) == len(result)

    cash, equity, shares, prev_weight = 100_000.0, 100_000.0, 0.0, 0.0
    expected_equity = []
    for price, weight in zip(prices, weights):
        trade_size = (weight - prev_weight) * equity
        cash -= trade_size
        shares += trade_size / price
        equity = cash + shares * price
        expected_equity.append(equity)
        prev_weight = weight

    assert result['equity'].tolist() == pytest.approx(expected_equity)


def test_transaction_costs(monkeypatch):
    """
    Same isolation strategy as test_pnl_accuracy (predict_rv/calc_weights faked
    to a known weight sequence) but this time with the real, nonzero
    transaction_cost_rate left in — the cost deduction itself is under test.
    Day 2 repeats day 1's weight, so trade_size is 0 that day: execute_trade
    short-circuits on trade_size == 0 (portfolio.py:27-28), which must skip
    the cost as well as the trade, not charge a cost with no matching trade.
    """
    from lib.backtester import engine
    from lib.backtester.portfolio import Portfolio

    weights = [0.4, 0.4, -0.3, 0.5]
    cost_rate = 0.000945  # engine.run_backtest's own default

    weight_iter = iter(weights)
    monkeypatch.setattr(engine, "predict_rv", lambda *args, **kwargs: next(weight_iter))
    monkeypatch.setattr(engine, "calc_weights", lambda forecast: forecast)
    monkeypatch.setattr(engine, "get_bars", make_fake_bars)
    monkeypatch.setattr(engine, "get_rate_and_close", make_fake_rates)

    portfolio = Portfolio(cash=100_000.0, shares=0.0, equity=100_000.0, peak_equity=100_000.0, cumulative_pnl=0.0)
    result = engine.run_backtest(portfolio, train_years=1, start="2020-06-01", end="2020-06-04")

    dates = list(pd.bdate_range("2020-06-01", "2020-06-04").date)
    prices = [daily_close(d) for d in dates]
    assert len(prices) == len(weights) == len(result)

    cash, equity, shares, prev_weight = 100_000.0, 100_000.0, 0.0, 0.0
    expected_equity = []
    for price, weight in zip(prices, weights):
        trade_size = (weight - prev_weight) * equity
        if trade_size != 0:
            cash -= trade_size + cost_rate * abs(trade_size)
            shares += trade_size / price
        equity = cash + shares * price
        expected_equity.append(equity)
        prev_weight = weight

    assert result['equity'].tolist() == pytest.approx(expected_equity)


def make_gapped_rates(dropped_date):
    """Returns a get_rate_and_close replacement with one date removed, simulating
    a data-vendor gap on that day."""
    def _make(symbol: str, start: str, end: str) -> pd.DataFrame:
        df = make_fake_rates(symbol, start, end)
        return df[df['timestamp'].dt.date != dropped_date].reset_index(drop=True)
    return _make


def test_handles_data_gap(monkeypatch):
    """
    Drops one business day from the middle of the rates/close fixture and
    confirms the engine skips it cleanly instead of crashing, without
    resetting prev_weight across the gap — the trade sized on the day *after*
    the gap must be relative to the weight held going into the gap (0.4, from
    the day before), not relative to 0.
    """
    from lib.backtester import engine
    from lib.backtester.portfolio import Portfolio

    all_dates = list(pd.bdate_range("2020-06-01", "2020-06-04").date)
    dropped_date = all_dates[1]  # 2020-06-02
    surviving_dates = [d for d in all_dates if d != dropped_date]

    weights = [0.4, -0.3, 0.5]  # one weight per surviving day
    assert len(weights) == len(surviving_dates)

    weight_iter = iter(weights)
    monkeypatch.setattr(engine, "predict_rv", lambda *args, **kwargs: next(weight_iter))
    monkeypatch.setattr(engine, "calc_weights", lambda forecast: forecast)
    monkeypatch.setattr(engine, "get_bars", make_fake_bars)
    monkeypatch.setattr(engine, "get_rate_and_close", make_gapped_rates(dropped_date))

    portfolio = Portfolio(cash=100_000.0, shares=0.0, equity=100_000.0, peak_equity=100_000.0, cumulative_pnl=0.0)
    result = engine.run_backtest(portfolio, train_years=1, start="2020-06-01", end="2020-06-04",
                                  transaction_cost_rate=0.0)

    assert result['timestamp'].dt.date.tolist() == surviving_dates

    prices = [daily_close(d) for d in surviving_dates]
    cash, equity, shares, prev_weight = 100_000.0, 100_000.0, 0.0, 0.0
    expected_equity = []
    for price, weight in zip(prices, weights):
        trade_size = (weight - prev_weight) * equity
        cash -= trade_size
        shares += trade_size / price
        equity = cash + shares * price
        expected_equity.append(equity)
        prev_weight = weight

    assert result['equity'].tolist() == pytest.approx(expected_equity)


def test_single_day_backtest(monkeypatch):
    """
    Degenerate case: start == end, so the event loop body runs exactly once.
    Exercises the loop's boundary behavior (first iteration is also the last)
    rather than the steady-state multi-day behavior the other tests cover.
    With transaction_cost_rate=0 and prev_weight starting at 0.0, a trade
    executed at day 0's own close can't move equity (cash -= trade_size,
    shares*price += trade_size, and those cancel exactly) — so equity after
    the only day must be exactly 100_000.0, an easy, sharp invariant for a
    single-day run.
    """
    from lib.backtester import engine
    from lib.backtester.portfolio import Portfolio

    day = "2020-06-01"
    weight = 0.5

    monkeypatch.setattr(engine, "predict_rv", lambda *args, **kwargs: weight)
    monkeypatch.setattr(engine, "calc_weights", lambda forecast: forecast)
    monkeypatch.setattr(engine, "get_bars", make_fake_bars)
    monkeypatch.setattr(engine, "get_rate_and_close", make_fake_rates)

    portfolio = Portfolio(cash=100_000.0, shares=0.0, equity=100_000.0, peak_equity=100_000.0, cumulative_pnl=0.0)
    result = engine.run_backtest(portfolio, train_years=1, start=day, end=day, transaction_cost_rate=0.0)

    assert len(result) == 1
    assert result['weight'].iloc[0] == pytest.approx(weight)
    assert result['equity'].iloc[0] == pytest.approx(100_000.0)


def test_fixture_smoke(monkeypatch):
    """
    Not a real engine test — just confirms the fixture itself is usable end to
    end (enough history for retrain_model's window check, no singular-matrix
    crash from OLS, no missing-date crash from predict_rv) before it's built on.
    Also the reference pattern for how real test cases should monkeypatch:
    patch the names as imported into engine.py, not the originals in
    data_utils.py.
    """
    from lib.backtester import engine
    from lib.backtester.portfolio import Portfolio

    monkeypatch.setattr(engine, "get_bars", make_fake_bars)
    monkeypatch.setattr(engine, "get_rate_and_close", make_fake_rates)

    portfolio = Portfolio(cash=100_000, shares=0, equity=100_000, peak_equity=100_000, cumulative_pnl=0)
    result = engine.run_backtest(portfolio, train_years=1, start="2020-06-01", end="2020-06-10")

    assert len(result) > 0
    assert not result.isna().any().any()
