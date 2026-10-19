"""CREATE TABLE statements and a function to initialise the database."""
import sqlite3

BARS_TABLE = """
CREATE TABLE IF NOT EXISTS bars (
    symbol       TEXT    NOT NULL,
    timeframe    TEXT    NOT NULL,
    time         INTEGER NOT NULL,  -- bar open time, UTC unix seconds
    open         REAL    NOT NULL,
    high         REAL    NOT NULL,
    low          REAL    NOT NULL,
    close        REAL    NOT NULL,
    tick_volume  INTEGER NOT NULL,
    spread       INTEGER NOT NULL,  -- in points
    real_volume  INTEGER NOT NULL,
    PRIMARY KEY (symbol, timeframe, time)
) WITHOUT ROWID;
"""

RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,              -- UTC, ISO 8601
    name       TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    settings   TEXT NOT NULL,              -- JSON
    summary    TEXT NOT NULL DEFAULT '{}', -- JSON: aggregate and backtest numbers
    notes      TEXT NOT NULL DEFAULT ''
);
"""

FOLD_METRICS_TABLE = """
CREATE TABLE IF NOT EXISTS fold_metrics (
    run_id                     INTEGER NOT NULL REFERENCES runs(run_id),
    fold                       INTEGER NOT NULL,
    test_from                  TEXT    NOT NULL,
    test_to                    TEXT    NOT NULL,
    test_direction_accuracy    REAL    NOT NULL,
    baseline_majority_accuracy REAL    NOT NULL,
    test_rmse                  REAL    NOT NULL,
    baseline_zero_rmse         REAL    NOT NULL,
    test_rank_corr             REAL,
    best_epoch                 INTEGER NOT NULL,
    n_test                     INTEGER NOT NULL,
    PRIMARY KEY (run_id, fold)
) WITHOUT ROWID;
"""


def init_db(conn: sqlite3.Connection) -> None:
    for statement in (BARS_TABLE, RUNS_TABLE, FOLD_METRICS_TABLE):
        conn.execute(statement)
    conn.commit()
