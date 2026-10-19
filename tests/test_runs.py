"""Tests for experiment logging in SQLite (temporary database)."""
import pandas as pd
import pytest

from fxml.db.connection import connect
from fxml.db.repository import load_fold_metrics, load_runs, save_run
from fxml.db.schema import init_db


@pytest.fixture
def conn(tmp_path):
    connection = connect(tmp_path / "test.db")
    init_db(connection)
    yield connection
    connection.close()


def fold_metrics():
    return pd.DataFrame(
        {
            "test_from": pd.to_datetime(["2020-01-01", "2020-07-01"], utc=True),
            "test_to": pd.to_datetime(["2020-06-30", "2020-12-31"], utc=True),
            "test_direction_accuracy": [0.52, 0.51],
            "baseline_majority_accuracy": [0.49, 0.50],
            "test_rmse": [0.002, 0.0021],
            "baseline_zero_rmse": [0.0021, 0.0021],
            "test_rank_corr": [0.09, float("nan")],
            "best_epoch": [2, 4],
            "n_test": [3100, 3100],
        },
        index=pd.Index([1, 2], name="fold"),
    )


def test_save_run_returns_increasing_ids(conn):
    first = save_run(conn, "a", "EURUSD", "H1", {"horizon": 4}, fold_metrics())
    second = save_run(conn, "b", "EURUSD", "H1", {"horizon": 12}, fold_metrics())
    assert second == first + 1


def test_settings_and_summary_roundtrip_as_dicts(conn):
    save_run(conn, "a", "EURUSD", "H1", {"horizon": 4, "lr": 0.001}, fold_metrics(),
             summary={"net_bps": 0.06}, notes="baseline")
    row = load_runs(conn).iloc[0]
    assert row["settings"] == {"horizon": 4, "lr": 0.001}
    assert row["summary"] == {"net_bps": 0.06}
    assert row["notes"] == "baseline"
    assert row["name"] == "a"


def test_fold_metrics_roundtrip(conn):
    run_id = save_run(conn, "a", "EURUSD", "H1", {}, fold_metrics())
    loaded = load_fold_metrics(conn, run_id)
    assert loaded.index.tolist() == [1, 2]
    assert loaded["test_direction_accuracy"].tolist() == [0.52, 0.51]
    assert loaded["best_epoch"].tolist() == [2, 4]
    assert pd.isna(loaded["test_rank_corr"].iloc[1])
    assert loaded["test_from"].iloc[0] == pd.Timestamp("2020-01-01", tz="UTC")


def test_load_runs_lists_newest_first(conn):
    save_run(conn, "old", "EURUSD", "H1", {}, fold_metrics())
    save_run(conn, "new", "EURUSD", "H1", {}, fold_metrics())
    assert load_runs(conn)["name"].tolist() == ["new", "old"]


def test_init_is_safe_to_repeat(conn):
    init_db(conn)
    init_db(conn)
    save_run(conn, "a", "EURUSD", "H1", {}, fold_metrics())
    assert len(load_runs(conn)) == 1
