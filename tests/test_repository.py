"""Tests for the SQLite repository (uses a temporary database, never forex.db)."""
import pandas as pd
import pytest

from fxml.db.connection import connect
from fxml.db.repository import load_bars, save_bars
from fxml.db.schema import init_db


@pytest.fixture
def conn(tmp_path):
    connection = connect(tmp_path / "test.db")
    init_db(connection)
    yield connection
    connection.close()


def make_bars(closes=(1.10, 1.20, 1.30)):
    index = pd.date_range("2024-01-01 00:00", periods=3, freq="h", tz="UTC", name="time")
    return pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.5, 1.5, 1.5],
            "low": [0.9, 0.9, 0.9],
            "close": list(closes),
            "tick_volume": [10, 20, 30],
            "spread": [1, 1, 1],
            "real_volume": [0, 0, 0],
        },
        index=index,
    )


def test_roundtrip_preserves_times_and_values(conn):
    bars = make_bars()
    save_bars(conn, "EURUSD", "H1", bars)
    loaded = load_bars(conn, "EURUSD", "H1")
    assert list(loaded.index) == list(bars.index)
    assert loaded["close"].tolist() == [1.10, 1.20, 1.30]
    assert loaded["tick_volume"].tolist() == [10, 20, 30]


def test_saving_twice_does_not_duplicate(conn):
    save_bars(conn, "EURUSD", "H1", make_bars())
    save_bars(conn, "EURUSD", "H1", make_bars())
    assert len(load_bars(conn, "EURUSD", "H1")) == 3


def test_saving_again_replaces_changed_values(conn):
    save_bars(conn, "EURUSD", "H1", make_bars())
    save_bars(conn, "EURUSD", "H1", make_bars(closes=(9.0, 9.0, 9.0)))
    assert load_bars(conn, "EURUSD", "H1")["close"].tolist() == [9.0, 9.0, 9.0]


def test_start_is_inclusive_and_end_is_exclusive(conn):
    save_bars(conn, "EURUSD", "H1", make_bars())
    loaded = load_bars(conn, "EURUSD", "H1",
                       start="2024-01-01 01:00", end="2024-01-01 02:00")
    assert loaded["close"].tolist() == [1.20]


def test_symbols_and_timeframes_are_separate(conn):
    save_bars(conn, "EURUSD", "H1", make_bars())
    assert load_bars(conn, "GBPUSD", "H1").empty
    assert load_bars(conn, "EURUSD", "M15").empty
