"""
Translates HAR-RV forecasts into trade decisions. This is where strategy logic
lives — e.g., go long vol when forecast is low, reduce exposure when forecast is
high. Kept separate from engine.py so different strategies can be plugged in
without touching the backtesting infrastructure.
"""

import os

from dotenv import load_dotenv
load_dotenv()

target_vol = float(os.getenv("target_vol"))
max_leverage = float(os.getenv("max_leverage"))

def calc_weights(forecast: float) -> float:

   annualized_vol = (forecast * 252) ** 0.5 if forecast > 0. else 0.
   weight = (target_vol / annualized_vol) if annualized_vol > 0. else 0.

   return min(weight, max_leverage);
