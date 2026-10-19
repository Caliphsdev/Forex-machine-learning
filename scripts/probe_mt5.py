"""Throwaway: ask the MT5 terminal what account, symbols and history it has."""
from datetime import datetime, timezone

from fxml.ingest.mt5_client import mt5_session

MAX_BARS = 2000000


def fmt(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    with mt5_session() as mt5:
        info = mt5.account_info()
        print("Server :", info.server)
        print("Company:", info.company)
        print("Demo   :", info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO)

        names = [s.name for s in mt5.symbols_get("*EURUSD*")]
        print("EURUSD-like symbols:", names)
        if not names:
            return

        symbol = names[0]
        mt5.symbol_select(symbol, True)
        timeframes = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
        }
        for label, tf in timeframes.items():
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, MAX_BARS)
            if rates is None or len(rates) == 0:
                print(f"{symbol} {label}: no data, error {mt5.last_error()}")
                continue
            print(f"{symbol} {label}: {len(rates):>7} bars  "
                  f"{fmt(rates[0]['time'])}  ->  {fmt(rates[-1]['time'])}")


if __name__ == "__main__":
    main()
