"""
Unit tests for performance metrics. Verifies: Sharpe of a flat return series is 0,
max drawdown of a monotonically increasing series is 0, Sortino handles zero
downside variance. Pure function tests — easy to write, high value since metrics
bugs silently make bad strategies look good.
"""
