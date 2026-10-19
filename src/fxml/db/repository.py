"""Read/write functions: save bars, load bars. No SQL anywhere else."""
import sqlite3
import json
import pandas as pd
from datetime import datetime, timezone


BAR_COLUMNS = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _to_epoch(value) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


def save_bars(conn: sqlite3.Connection, symbol: str, timeframe: str,
              bars: pd.DataFrame) -> int:
    """Insert or replace bars. `bars` has a UTC DatetimeIndex named 'time'
    and the BAR_COLUMNS. Returns the number of rows written."""
    df = bars.reset_index()
    epochs = ((df["time"] - _EPOCH) // pd.Timedelta(seconds=1)).tolist()
    columns = [df[name].tolist() for name in BAR_COLUMNS]
    rows = [
        (symbol, timeframe, *values)
        for values in zip(epochs, *columns)
    ]
    sql = (
        "INSERT OR REPLACE INTO bars "
        "(symbol, timeframe, time, open, high, low, close, "
        "tick_volume, spread, real_volume) VALUES (?,?,?,?,?,?,?,?,?,?)"
    )
    with conn:
        conn.executemany(sql, rows)
    return len(rows)


def load_bars(conn: sqlite3.Connection, symbol: str, timeframe: str,
              start=None, end=None) -> pd.DataFrame:
    """Load bars as a DataFrame indexed by UTC time. start inclusive, end exclusive."""
    query = (
        "SELECT time, open, high, low, close, tick_volume, spread, real_volume "
        "FROM bars WHERE symbol = ? AND timeframe = ?"
    )
    params: list = [symbol, timeframe]
    if start is not None:
        query += " AND time >= ?"
        params.append(_to_epoch(start))
    if end is not None:
        query += " AND time < ?"
        params.append(_to_epoch(end))
    query += " ORDER BY time"

    df = pd.read_sql_query(query, conn, params=params)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.set_index("time")


METRIC_COLUMNS = [
    "test_direction_accuracy", "baseline_majority_accuracy", "test_rmse",
    "baseline_zero_rmse", "test_rank_corr", "best_epoch", "n_test",
]


def save_run(conn: sqlite3.Connection, name: str, symbol: str, timeframe: str,
             settings: dict, fold_metrics: pd.DataFrame,
             summary: dict | None = None, notes: str = "") -> int:
    """Store one walk-forward run and its per-fold metrics; returns the run id.

    `fold_metrics` is indexed by fold number and has test_from, test_to and METRIC_COLUMNS.
    """
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with conn:
        cursor = conn.execute(
            "INSERT INTO runs (created_at, name, symbol, timeframe, settings, summary, notes) "
            "VALUES (?,?,?,?,?,?,?)",
            (created_at, name, symbol, timeframe, json.dumps(settings, sort_keys=True),
             json.dumps(summary or {}, sort_keys=True, default=float), notes),
        )
        run_id = cursor.lastrowid
        rows = [
            (run_id, int(fold), pd.Timestamp(row["test_from"]).isoformat(),
             pd.Timestamp(row["test_to"]).isoformat(),
             *(None if pd.isna(row[c]) else row[c].item() if hasattr(row[c], "item") else row[c]
               for c in METRIC_COLUMNS))
            for fold, row in fold_metrics.iterrows()
        ]
        conn.executemany(
            "INSERT INTO fold_metrics (run_id, fold, test_from, test_to, "
            + ", ".join(METRIC_COLUMNS) + ") VALUES (?,?,?,?,"
            + ",".join("?" * len(METRIC_COLUMNS)) + ")",
            rows,
        )
    return run_id


def load_runs(conn: sqlite3.Connection) -> pd.DataFrame:
    """All runs, newest first, with settings and summary decoded from JSON."""
    df = pd.read_sql_query("SELECT * FROM runs ORDER BY run_id DESC", conn)
    df["settings"] = df["settings"].map(json.loads)
    df["summary"] = df["summary"].map(json.loads)
    return df.set_index("run_id")


def load_fold_metrics(conn: sqlite3.Connection, run_id: int) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT * FROM fold_metrics WHERE run_id = ? ORDER BY fold", conn, params=[run_id])
    df["test_from"] = pd.to_datetime(df["test_from"])
    df["test_to"] = pd.to_datetime(df["test_to"])
    return df.drop(columns="run_id").set_index("fold")
