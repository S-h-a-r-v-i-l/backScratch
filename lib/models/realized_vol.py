"""
RV estimators: compute realized volatility from raw intraday price data.
Includes standard RV (sum of squared intraday returns), bipower variation
(jump-robust), and kernel-based estimators. Kept separate from har_rv.py so
estimators can be swapped without touching the model and tested independently.
"""
