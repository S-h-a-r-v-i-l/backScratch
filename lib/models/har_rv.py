"""
HAR-RV model (Corsi 2009). Regresses future realized volatility on three lagged RV
components: daily (RV_d), weekly (RV_w = mean of last 5 days), and monthly
(RV_m = mean of last 22 days). Fits via OLS, produces point forecasts and optionally
confidence intervals.
"""
