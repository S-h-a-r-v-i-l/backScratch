"""
Tests the backtesting engine's integrity. Verifies no look-ahead bias (engine at
time T cannot access prices at T+1), P&L accumulates correctly, transaction costs
apply properly, and the event loop handles edge cases (gaps, early termination).
"""
