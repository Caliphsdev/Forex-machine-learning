"""Orchestrates fetching bars for many symbols/timeframes and storing them via the repository."""
import sqlite3

import pandas as pd

from fxml import config
from fxml.db.repository import BAR_COLUMNS, save_bars
from fxml.ingest.mt5_client import fetch_rates, mt5_session
from fxml.utils.timeutils import server_to_utc


def rates_to_bars(rates) -> pd.DataFrame:
    """Raw MT5 rates array -> DataFrame indexed by true UTC time."""
    df = pd.DataFrame(rates)
    server_time = pd.to_datetime(df["time"], unit="s")  # naive broker wall clock
    df["time"] = server_to_utc(server_time)
    return df.set_index("time")[BAR_COLUMNS]


def download_all(conn: sqlite3.Connection) -> None:
    """Download every configured symbol/timeframe and upsert into SQLite."""
    start = pd.Timestamp(config.HISTORY_START, tz="UTC")
    with mt5_session():
        for symbol in config.SYMBOLS:
            for timeframe in config.TIMEFRAMES:
                rates = fetch_rates(symbol, timeframe, config.MAX_BARS)
                # The last bar is still forming; storing it would feed the model
                # a half-finished candle. It gets saved on the next run.
                bars = rates_to_bars(rates[:-1])
                bars = bars[bars.index >= start]
                written = save_bars(conn, symbol, timeframe, bars)
                print(f"{symbol} {timeframe}: {written:>7} bars  "
                      f"{bars.index[0]:%Y-%m-%d %H:%M} -> {bars.index[-1]:%Y-%m-%d %H:%M} UTC")
