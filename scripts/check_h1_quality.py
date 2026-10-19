"""Throwaway: check H1 data density per year and the broker's weekly open/close hours."""
import pandas as pd

from fxml.ingest.mt5_client import mt5_session


def main() -> None:
    with mt5_session() as mt5:
        mt5.symbol_select("EURUSD", True)
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 500_000)

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")  # naive: broker's server clock

    print("H1 bars per year (a full year is roughly 6,200):")
    print(df.groupby(df["time"].dt.year).size().tail(30).to_string())

    recent = df[df["time"] >= "2018-01-01"].copy()
    recent["week"] = recent["time"].dt.to_period("W")
    first = recent.groupby("week")["time"].min()
    last = recent.groupby("week")["time"].max()

    print("\nFirst bar of each week (weekday 0=Mon, hour):")
    print(first.groupby([first.dt.weekday, first.dt.hour]).size().to_string())
    print("\nLast bar of each week (weekday 4=Fri, hour):")
    print(last.groupby([last.dt.weekday, last.dt.hour]).size().to_string())


if __name__ == "__main__":
    main()
