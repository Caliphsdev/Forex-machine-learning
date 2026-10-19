"""Timezone helpers: convert MT5 broker server time to UTC."""
"""Timezone helpers: convert MT5 broker server time to UTC."""
import pandas as pd

NEW_YORK = "America/New_York"
SERVER_AHEAD_OF_NY = pd.Timedelta(hours=7)


def server_to_utc(server_times: pd.Series) -> pd.Series:
    """Convert broker server wall-clock times (naive) to tz-aware UTC.

    The broker clock is New York time + 7h, so subtract 7h to get New York
    wall time, then let the tz database apply the right daylight-saving rules.
    Ambiguous or non-existent times raise an error instead of guessing.
    """
    ny_wall = server_times - SERVER_AHEAD_OF_NY
    return ny_wall.dt.tz_localize(NEW_YORK).dt.tz_convert("UTC")
