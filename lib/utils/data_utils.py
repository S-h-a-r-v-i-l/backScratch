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

def get_fred_data(series_id: str, start: str, end: str) -> pd.DataFrame:
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={start}&observation_end={end}"
    response = requests.get(url)
    data = response.json()
    observations = data['observations']
    df = pd.DataFrame(observations)
    df['date'] = pd.to_datetime(df['date'])
    df['rate'] = pd.to_numeric(df['value'], errors='coerce')
    df = df[['date', 'rate']].rename(columns={'date': 'timestamp'})
    return df


def load_bars(symbol: str, start: str, end: str, timeframe_minutes: int = 5, timeframe_unit: TimeFrameUnit = TimeFrameUnit.Minute)-> pd.DataFrame:
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
        timeframe=TimeFrame(timeframe_minutes, timeframe_unit),
        start=start,
        limit=None,
        end=end,
        adjustment="raw"
    )

    bars = client.get_stock_bars(request)
    df = bars.df 
    return df.reset_index()[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def save_as_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path)

def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def get_bars(symbol: str, start: str, end: str, timeframe_minutes: int = 5) -> pd.DataFrame:
    """
    Load bars from data or API. Returns a DataFrame with columns:
    ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    """
    parquet_path = f"data/processed/{symbol}_{start}_{end}_{timeframe_minutes}min.parquet"
    
    if os.path.exists(parquet_path):
        print(f"Loaded bars from {parquet_path}")
        return load_parquet(parquet_path)
        
    
    df = load_bars(symbol, start, end, timeframe_minutes)
    save_as_parquet(df, parquet_path)
    return df

def get_daily_close(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Load daily close prices from data or API. Returns a DataFrame with columns:
    ['timestamp', 'close'].
    """
    parquet_path = f"data/processed/{symbol}_{start}_{end}_daily_close.parquet"
    
    if os.path.exists(parquet_path):
        print(f"Loaded daily close from {parquet_path}")
        return load_parquet(parquet_path)
        
    
    df = load_bars(symbol, start, end, timeframe_minutes=1, timeframe_unit=TimeFrameUnit.Day)
    df = df[['timestamp', 'close']]
    save_as_parquet(df, parquet_path)
    return df

def get_rate_and_close(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Load daily close prices and daily cash/borrow rates from FRED (DGS3MO +/- 1.5%
    spread). Returns a DataFrame with columns:
    ['timestamp', 'close', 'cash_rate', 'borrow_rate', 'risk_free_rate'].
    """
    close_df = get_daily_close(symbol, start, end)
    rate_df = get_fred_data(series_id="DGS3MO", start=start, end=end)

    close_df['timestamp'] = close_df['timestamp'].dt.tz_convert(None).dt.normalize()

    df = pd.merge(close_df, rate_df, on='timestamp', how='left')
    df['rate'] = df['rate'].ffill()

    df['cash_rate'] = (1 + (df['rate'] - 1.5) / 100) ** (1/252) - 1
    df['borrow_rate'] = (1 + (df['rate'] + 1.5) / 100) ** (1/252) - 1
    df['risk_free_rate'] = (1 + df['rate'] / 100) ** (1/252) - 1

    return df[['timestamp', 'close', 'cash_rate', 'borrow_rate', 'risk_free_rate']]

if __name__ == "__main__":
    df = get_fred_data(series_id="DGS3MO", start="2024-01-01", end="2025-01-01")
    print(df)
