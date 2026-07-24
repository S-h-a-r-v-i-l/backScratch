"""
Core backtesting event loop. Walks forward through time bar-by-bar, calls signal
logic, updates positions, and records state. Enforces no look-ahead bias: only
data up to the current timestep is exposed to downstream code at any point.
"""
