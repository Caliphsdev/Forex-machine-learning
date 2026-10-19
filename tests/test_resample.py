"""Tests for resampling H1 bars into FX trading days."""
import pandas as pd

from fxml.features.resample import to_daily


def h1_bars(start: str, n: int) -> pd.DataFrame:
    """n consecutive H1 bars from `start` (UTC); bar i opens at i and closes at i + 0.25."""
    index = pd.date_range(start, periods=n, freq="1h", tz="UTC", name="time")
    i = pd.Series(range(n), index=index, dtype=float)
    return pd.DataFrame({
        "open": i, "high": i + 0.5, "low": i - 0.5, "close": i + 0.25,
        "tick_volume": 10, "spread": 8, "real_volume": 0,
    })


def test_a_fx_day_becomes_one_daily_bar_with_the_right_ohlc():
    # 17:00 New York on 2 Jan 2024 (winter, UTC-5) is 22:00 UTC
    bars = h1_bars("2024-01-02 22:00", 48)  # two full FX days
    daily = to_daily(bars)
    assert len(daily) == 2
    first = daily.iloc[0]
    assert first["open"] == 0.0
    assert first["high"] == 23.5
    assert first["low"] == -0.5
    assert first["close"] == 23.25
    assert first["tick_volume"] == 240
    assert first["spread"] == 8
    assert first["n_bars"] == 24
    assert daily.index[0] == bars.index[0]


def test_the_day_boundary_is_5pm_new_york_in_summer_too():
    # in July New York is UTC-4, so 17:00 New York is 21:00 UTC
    bars = h1_bars("2024-07-02 19:00", 4)  # 19:00 and 20:00 | 21:00 and 22:00 UTC
    daily = to_daily(bars)
    assert list(daily["n_bars"]) == [2, 2]
    assert daily.index[1] == pd.Timestamp("2024-07-02 21:00", tz="UTC")


def test_finished_days_do_not_change_when_later_bars_arrive():
    bars = h1_bars("2024-01-02 22:00", 24 * 5)
    cut = bars.iloc[: 24 * 3 + 5]  # stops part-way through the fourth day
    pd.testing.assert_frame_equal(to_daily(cut).iloc[:3], to_daily(bars).iloc[:3])


def test_a_half_finished_day_shows_how_few_bars_it_has():
    bars = h1_bars("2024-01-02 22:00", 24 * 3 + 5)
    assert to_daily(bars)["n_bars"].iloc[-1] == 5

def test_daily_bar_keeps_the_spread_of_its_first_bar():
    # the first bar of an FX day is the 17:00 New York roll, when spreads are widest
    bars = h1_bars("2024-01-02 22:00", 48)
    bars.loc[bars.index[0], "spread"] = 30
    bars.loc[bars.index[24], "spread"] = 50
    daily = to_daily(bars)
    assert list(daily["open_spread"]) == [30, 50]
