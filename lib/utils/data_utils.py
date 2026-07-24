"""
Data loading and preprocessing utilities. Handles loading CSVs/Parquet from data/,
resampling tick data to fixed bars, forward-filling gaps, aligning timestamps, and
handling trading calendar issues. Keeps model and backtester code free of pandas
boilerplate.
"""
