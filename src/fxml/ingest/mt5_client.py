"""Thin wrapper around the MetaTrader5 package: connect, fetch rates, disconnect."""
from contextlib import contextmanager

import MetaTrader5 as mt5



@contextmanager
def mt5_session():
    """ Connect to the running MT5 terminal, and always disconnect afterwards."""
    if not mt5.initialize():
        raise RuntimeError(f"Failed to initialize MT5: {mt5.last_error()}")
    try:
        yield mt5
    finally:
        mt5.shutdown()

TIMEFRAME_LABELS = ("M1", "M5", "M15", "M30", "H1", "H4", "D1")


def fetch_rates(symbol: str, timeframe: str, count: int):
    """Fetch the most recent `count` bars. Call inside `mt5_session()`."""
    if timeframe not in TIMEFRAME_LABELS:
        raise ValueError(f"Unknown timeframe {timeframe!r}; use one of {TIMEFRAME_LABELS}")
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Cannot select symbol {symbol!r}: {mt5.last_error()}")
    tf = getattr(mt5, f"TIMEFRAME_{timeframe}")
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data for {symbol} {timeframe}: {mt5.last_error()}")
    return rates
