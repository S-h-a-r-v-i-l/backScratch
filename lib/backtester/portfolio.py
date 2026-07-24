"""
Tracks positions, cash, and running P&L. Handles position sizing (e.g.,
vol-targeting based on HAR-RV forecasts), transaction costs, and mark-to-market.
Separate from engine.py so sizing rules can be swapped without rewriting the loop.
"""
