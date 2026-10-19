"""Tests for timezone conversion."""
"""Tests for timezone conversion."""
import pandas as pd

from fxml.utils.timeutils import server_to_utc


def _convert(text: str) -> pd.Timestamp:
    return server_to_utc(pd.Series(pd.to_datetime([text]))).iloc[0]


def test_winter_server_is_utc_plus_2():
    assert _convert("2024-01-15 10:00") == pd.Timestamp("2024-01-15 08:00", tz="UTC")


def test_summer_server_is_utc_plus_3():
    assert _convert("2024-07-15 10:00") == pd.Timestamp("2024-07-15 07:00", tz="UTC")


def test_winter_week_open_is_sunday_22_utc():
    assert _convert("2024-01-15 00:00") == pd.Timestamp("2024-01-14 22:00", tz="UTC")


def test_summer_week_open_is_sunday_21_utc():
    assert _convert("2024-07-15 00:00") == pd.Timestamp("2024-07-14 21:00", tz="UTC")
