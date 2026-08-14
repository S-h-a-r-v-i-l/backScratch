"""
Unit tests for performance metrics. Pure function tests — easy to write, high
value since metrics bugs silently make bad strategies look good.
"""

import numpy as np
import pandas as pd
import pytest

from lib.backtester.metrics import sharpe_ratio, sortino_ratio


def test_negative_mean_return():
    returns = pd.Series([-0.02, -0.01, 0.0, -0.03, -0.04])

    assert sharpe_ratio(returns) == pytest.approx(-20.079840636817814)
    assert sortino_ratio(returns) == pytest.approx(-12.96148139681572)

def test_zero_mean_return():
    returns = pd.Series([-0.02, 0.02, 0.0, -0.03, 0.03])

    assert sharpe_ratio(returns) == pytest.approx(0.0)
    assert sortino_ratio(returns) == pytest.approx(0.0)

def test_undefined_sharpe():
    returns = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])

    assert np.isnan(sharpe_ratio(returns))
    assert np.isnan(sortino_ratio(returns))

def test_zero_volatility():
    returns = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5])

    assert np.isnan(sharpe_ratio(returns))
    assert np.isnan(sortino_ratio(returns))

def test_default_case():
    # mix of up and down days so both Sharpe's std and Sortino's downside
    # deviation are well-defined, nonzero, and not near any edge case.
    returns = pd.Series([0.02, 0.03, -0.01, 0.01, -0.02])

    assert sharpe_ratio(returns) == pytest.approx(4.593220484431882)
    assert sortino_ratio(returns) == pytest.approx(9.524704719832526)
