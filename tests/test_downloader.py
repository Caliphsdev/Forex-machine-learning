"""Tests for converting raw MT5 rates into clean UTC bars."""
import numpy as np
import pandas as pd

from fxml.db.repository import BAR_COLUMNS
from fxml.ingest.downloader import rates_to_bars

RATES_DTYPE = [
    ("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"),
    ("close", "<f8"), ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8"),
]


def server_epoch(text: str) -> int:
    """MT5 gives server wall-clock time encoded as if it were UTC."""
    return int(pd.Timestamp(text, tz="UTC").timestamp())


def make_rates():
    return np.array(
        [
            (server_epoch("2024-01-15 10:00"), 1.0, 1.5, 0.9, 1.1, 10, 2, 0),
            (server_epoch("2024-07-15 10:00"), 2.0, 2.5, 1.9, 2.1, 20, 3, 0),
        ],
        dtype=RATES_DTYPE,
    )


def test_times_become_true_utc():
    bars = rates_to_bars(make_rates())
    assert bars.index[0] == pd.Timestamp("2024-01-15 08:00", tz="UTC")  # winter: +2h
    assert bars.index[1] == pd.Timestamp("2024-07-15 07:00", tz="UTC")  # summer: +3h


def test_index_is_named_time_and_columns_are_the_bar_columns():
    bars = rates_to_bars(make_rates())
    assert bars.index.name == "time"
    assert list(bars.columns) == BAR_COLUMNS


def test_values_are_kept():
    bars = rates_to_bars(make_rates())
    assert bars["close"].tolist() == [1.1, 2.1]
    assert bars["spread"].tolist() == [2, 3]
