"""
Translates HAR-RV forecasts into trade decisions. This is where strategy logic
lives — e.g., go long vol when forecast is low, reduce exposure when forecast is
high. Kept separate from engine.py so different strategies can be plugged in
without touching the backtesting infrastructure.
"""
