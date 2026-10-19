"""Throwaway: which pairs the broker offers, how deep the H1 history is, and what each costs."""
import pandas as pd

from fxml.ingest.mt5_client import mt5_session

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
         "EURGBP", "EURJPY", "GBPJPY", "EURCHF", "AUDJPY"]


def main() -> None:
    with mt5_session() as mt5:
        for symbol in PAIRS:
            info = mt5.symbol_info(symbol)
            if info is None or not mt5.symbol_select(symbol, True):
                print(f"{symbol:8} not available")
                continue
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 500_000)
            if rates is None or len(rates) == 0:
                print(f"{symbol:8} no H1 data")
                continue
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            recent = df[df["time"] >= df["time"].max() - pd.Timedelta(days=365)]
            cost_bps = (recent["spread"] * info.point / recent["close"]).median() * 1e4
            print(f"{symbol:8} {len(df):>7} bars from {df['time'].min():%Y-%m-%d}  "
                  f"median spread last year {cost_bps:.2f} bps  (digits {info.digits})")


if __name__ == "__main__":
    main()
