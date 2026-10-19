"""Turn H1 bars into FX trading days (a day runs 17:00 to 17:00 New York time)."""
import pandas as pd

from fxml.utils.timeutils import NEW_YORK, SERVER_AHEAD_OF_NY


def to_daily(bars: pd.DataFrame) -> pd.DataFrame:
    """Daily bars from H1 bars indexed by UTC time.

    A day starts at 17:00 New York, which is the broker's midnight and follows daylight saving.
    Each daily bar is stamped with the open time of its first H1 bar. `n_bars` counts the H1 bars
    inside, so a half-finished or holiday day can be spotted and dropped by the caller.
    `open_spread` is the spread of the day's first bar.
    """
    ny_wall = bars.index.tz_convert(NEW_YORK).tz_localize(None)
    day = (ny_wall + SERVER_AHEAD_OF_NY).normalize()  # the broker's calendar date
    daily = bars.groupby(day).agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), tick_volume=("tick_volume", "sum"),
            spread=("spread", "mean"), open_spread=("spread", "first"),
            n_bars=("close", "size"),
        )

    starts = bars.index.to_series().groupby(day).first()
    daily.index = pd.DatetimeIndex(starts, name="time")
    return daily


  
