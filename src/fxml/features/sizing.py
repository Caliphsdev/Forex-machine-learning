"""Position sizing: scale each pair's signal so every pair carries the same risk."""
import numpy as np
import pandas as pd

TRADING_DAYS = 260  # FX days per year in our daily bars (about 7,250 bars over 27.8 years)
VOL_SPAN = 60  # span of the exponentially weighted volatility estimate, in days
PAIR_VOL_TARGET = 0.10  # annual volatility each pair is sized to carry
PORTFOLIO_VOL_TARGET = 0.10  # annual volatility the whole portfolio is scaled to


def return_vol(returns: pd.Series, span: int = VOL_SPAN) -> pd.Series:
    """Exponentially weighted volatility of a return series; uses returns up to each day only."""
    return np.sqrt((returns ** 2).ewm(span=span, min_periods=span).mean())


def daily_vol(close: pd.Series, span: int = VOL_SPAN) -> pd.Series:
    """Exponentially weighted daily volatility of log returns; uses closes up to each day only."""
    return return_vol(np.log(close).diff(), span)


def risk_scaled_position(signal: pd.Series, close: pd.Series,
                         target: float = PAIR_VOL_TARGET, span: int = VOL_SPAN) -> pd.Series:
    """Exposure as a multiple of capital: signal times the risk target over annual volatility."""
    annual_vol = daily_vol(close, span) * np.sqrt(TRADING_DAYS)
    return signal * target / annual_vol


def vol_target_multiplier(returns: pd.Series, target: float = PORTFOLIO_VOL_TARGET,
                          span: int = VOL_SPAN) -> pd.Series:
    """Factor that scales a portfolio's daily returns to the target annual volatility.

    Built only from returns up to each day, so it can be applied to the next day's positions.
    """
    return target / (return_vol(returns, span) * np.sqrt(TRADING_DAYS))
