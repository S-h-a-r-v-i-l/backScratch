"""
Data loading and preprocessing utilities. Handles loading CSVs/Parquet from data/,
resampling tick data to fixed bars, forward-filling gaps, aligning timestamps, and
handling trading calendar issues. Keeps model and backtester code free of pandas
boilerplate.
"""
import pandas as pd
import requests
from dotenv import load_dotenv
import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.timeframe import TimeFrameUnit
def load_bars(symbol: str, start: str, end: str, timeframe_minutes: int = 5)-> pd.DataFrame:
    """
    Load bars from data. Returns a DataFrame with columns:
    ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    """
    
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY") 
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    client = StockHistoricalDataClient(api_key, secret_key)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(timeframe_minutes, TimeFrameUnit.Minute),
        start=start,
        end=end,
        limit=100_000,
        adjustment="raw"
    )

    bars = client.get_stock_bars(request)
    df = bars.df 
    return df.reset_index()[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def save_bars_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path)

def load_bars_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def get_bars(symbol: str, start: str, end: str, timeframe_minutes: int = 5) -> pd.DataFrame:
    """
    Load bars from data or API. Returns a DataFrame with columns:
    ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    """
    parquet_path = f"data/processed/{symbol}_{start}_{end}_{timeframe_minutes}min.parquet"
    
    if os.path.exists(parquet_path):
        print(f"Loaded bars from {parquet_path}")
        return load_bars_parquet(parquet_path)
        
    
    df = load_bars(symbol, start, end, timeframe_minutes)
    save_bars_parquet(df, parquet_path)
    return df



if __name__ == "__main__":
    df = get_bars(symbol="SPY", start="2024-01-01", end="2024-01-31")
    print(df.head())
