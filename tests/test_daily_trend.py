"""Tests for the pre-registered daily trend run: period handling, scoring and logging."""
import numpy as np
import pandas as pd
import pytest

from fxml.db.connection import connect
from fxml.db.repository import load_runs, save_bars
from fxml.db.schema import init_db
from fxml.training.daily_trend import (complete_days, load_daily_pairs, provenance,
                                       run_period, summarise)
from tests.test_resample import h1_bars

POINTS = {"AAA": 0.0001, "BBB": 0.0001, "CCC": 0.0001}
RESAMPLES = 200  # fewer bootstrap resamples keep the tests quick


@pytest.fixture
def conn(tmp_path):
    connection = connect(tmp_path / "test.db")
    init_db(connection)
    yield connection
    connection.close()


def synthetic_pair(seed: int) -> pd.DataFrame:
    """Business days from 1999 to 2017 with a mild upward drift, 24 H1 bars per day."""
    steps = np.random.default_rng(seed).normal(0.0003, 0.006, 4900)
    close = 1.2 * np.exp(np.cumsum(steps))
    index = pd.date_range("1999-01-01", periods=len(close), freq="B", tz="UTC", name="time")
    return pd.DataFrame({"open": np.r_[close[0], close[:-1]], "close": close,
                         "open_spread": 10.0, "n_bars": 24}, index=index)


def synthetic_pairs() -> dict:
    return {name: synthetic_pair(seed) for seed, name in enumerate(POINTS)}


def test_only_an_incomplete_final_day_is_dropped():
    daily = pd.DataFrame({"n_bars": [24, 5, 24, 11]},
                         index=pd.date_range("2024-01-01", periods=4, tz="UTC"))
    assert complete_days(daily)["n_bars"].tolist() == [24, 5, 24]  # the early thin day stays
    assert complete_days(daily.iloc[:3])["n_bars"].tolist() == [24, 5, 24]  # nothing to drop


def test_loading_builds_daily_bars_and_drops_the_unfinished_day(conn):
    save_bars(conn, "AAA", "H1", h1_bars("2024-01-02 22:00", 24 * 2 + 5))
    pairs = load_daily_pairs(conn, ["AAA"])
    assert len(pairs["AAA"]) == 2


def test_summarise_matches_the_hand_calculation():
    returns = pd.Series(np.random.default_rng(0).normal(0.0004, 0.006, 1000))
    stats = summarise(returns, resamples=RESAMPLES)
    assert stats["n_days"] == 1000
    assert stats["sharpe"] == pytest.approx(returns.mean() / returns.std() * np.sqrt(260))
    assert stats["annual_return"] == pytest.approx(returns.mean() * 260)
    assert stats["annual_vol"] == pytest.approx(returns.std() * np.sqrt(260))
    assert stats["max_drawdown"] == pytest.approx((returns.cumsum() - returns.cumsum().cummax()).min())
    assert stats["sharpe_ci90_low"] < stats["sharpe"] < stats["sharpe_ci90_high"]


def test_a_development_run_is_logged_with_its_verdict_and_provenance(conn):
    info = {"git_commit": "abc123", "git_dirty": False}
    result = run_period(conn, synthetic_pairs(), POINTS, "dev", info, resamples=RESAMPLES)
    row = load_runs(conn).iloc[0]
    assert row["name"] == "daily-trend-dev"
    assert row["settings"]["git_commit"] == "abc123"
    assert row["settings"]["symbols"] == ["AAA", "BBB", "CCC"]
    assert row["summary"]["verdict"] == result["verdict"]
    assert set(result["verdict"]) == {"sharpe", "drawdown", "vol_targeting", "all"}
    assert set(result) >= {"targeted", "baseline", "yearly_net", "per_pair_sharpe", "run_id"}


def test_a_development_run_only_scores_2001_to_2014(conn):
    result = run_period(conn, synthetic_pairs(), POINTS, "dev", resamples=RESAMPLES)
    years = sorted(int(year) for year in result["yearly_net"])
    assert years == list(range(2001, 2015))


def test_development_results_cannot_depend_on_later_prices(conn):
    original = synthetic_pairs()
    changed = {}
    for name, bars in original.items():
        bars = bars.copy()
        later = bars.index >= pd.Timestamp("2015-02-01", tz="UTC")
        bars.loc[later, ["open", "close"]] *= 3.0  # prices after the development period change
        changed[name] = bars
    first = run_period(conn, original, POINTS, "dev", resamples=RESAMPLES)
    second = run_period(conn, changed, POINTS, "dev", resamples=RESAMPLES)
    assert first["targeted"] == second["targeted"]
    assert first["baseline"] == second["baseline"]


def test_a_confirmation_run_also_scores_doubled_costs(conn):
    result = run_period(conn, synthetic_pairs(), POINTS, "confirm", resamples=RESAMPLES)
    assert result["stressed"]["sharpe"] < result["normal"]["sharpe"]
    assert set(result["verdict"]) == {"sharpe", "stress", "drawdown", "all"}
    assert load_runs(conn).iloc[0]["name"] == "daily-trend-confirm"


def test_the_confirmation_run_can_only_be_done_once(conn):
    run_period(conn, synthetic_pairs(), POINTS, "confirm", resamples=RESAMPLES)
    with pytest.raises(RuntimeError):
        run_period(conn, synthetic_pairs(), POINTS, "confirm", resamples=RESAMPLES)
    assert len(load_runs(conn)) == 1


def test_an_unknown_period_is_rejected(conn):
    with pytest.raises(ValueError):
        run_period(conn, synthetic_pairs(), POINTS, "test", resamples=RESAMPLES)


def test_provenance_records_the_pass_mark_fingerprint_and_survives_no_git(tmp_path):
    marks = tmp_path / "marks.md"
    marks.write_text("pass marks")
    info = provenance(tmp_path, marks)
    assert len(info["preregistration_sha256"]) == 64
    assert set(info) == {"git_commit", "git_dirty", "preregistration_sha256"}
