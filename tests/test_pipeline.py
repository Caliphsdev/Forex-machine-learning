"""End-to-end test: stored bars -> walk-forward -> costed backtest -> logged run."""
import pytest
import torch

from fxml.db.connection import connect
from fxml.db.repository import load_fold_metrics, load_runs, save_bars
from fxml.db.schema import init_db
from fxml.training.experiment import Settings
from fxml.training.pipeline import ExperimentPlan, run_experiment, run_experiment_detailed
from tests.test_builder import make_bars

CPU = torch.device("cpu")
SETTINGS = Settings(lookback=8, horizon=4, hidden_size=8, val_size=60,
                    epochs=2, patience=2, batch_size=64)


@pytest.fixture
def conn(tmp_path):
    connection = connect(tmp_path / "pipeline.db")
    init_db(connection)
    save_bars(connection, "EURUSD", "H1", make_bars(1200))
    yield connection
    connection.close()


def plan(**overrides):
    values = dict(name="test-run", start="2024-01-01", train_size=300, test_size=100)
    return ExperimentPlan(**{**values, **overrides})


def test_full_run_is_saved_with_its_folds_and_summary(conn):
    summary = run_experiment(conn, plan(flat_cost_bps=0.8), SETTINGS, device=CPU, verbose=False)
    runs = load_runs(conn)
    assert runs.index.tolist() == [summary["run_id"]]
    assert runs.iloc[0]["name"] == "test-run"
    assert runs.iloc[0]["settings"]["plan"]["train_size"] == 300
    assert runs.iloc[0]["settings"]["settings"]["horizon"] == 4
    assert len(load_fold_metrics(conn, summary["run_id"])) == summary["n_folds"] >= 2
    assert "net_bps_with_0.3_slippage" in runs.iloc[0]["summary"]


def test_flat_cost_is_charged_exactly_on_every_trade(conn):
    summary = run_experiment(conn, plan(flat_cost_bps=0.8), SETTINGS, device=CPU, verbose=False)
    assert summary["n_trades"] > 0
    assert summary["gross_bps_per_trade"] - summary["net_bps_per_trade"] == pytest.approx(0.8)


def test_recorded_spread_is_used_when_no_flat_cost_is_given(conn):
    summary = run_experiment(conn, plan(), SETTINGS, device=CPU, verbose=False)
    cost_bps = summary["gross_bps_per_trade"] - summary["net_bps_per_trade"]
    assert 0.4 < cost_bps < 0.9  # 7 points = 0.00007 on a price near 1.1 is about 0.64 bps


def test_empty_date_range_raises(conn):
    with pytest.raises(ValueError):
        run_experiment(conn, plan(start="2030-01-01"), SETTINGS, device=CPU, verbose=False)

def test_each_symbol_uses_its_own_point_size_for_the_cost(conn):
    save_bars(conn, "USDJPY", "H1", make_bars(1200, scale=100.0))  # prices near 110, 7 points
    summary = run_experiment(conn, plan(symbol="USDJPY"), SETTINGS, device=CPU, verbose=False)
    cost_bps = summary["gross_bps_per_trade"] - summary["net_bps_per_trade"]
    assert 0.4 < cost_bps < 0.9  # 7 * 0.001 / 110 = 0.64 bps; a 1e-5 point would give 0.006


def test_symbol_without_a_point_size_raises(conn):
    save_bars(conn, "XYZABC", "H1", make_bars(1200))
    with pytest.raises(ValueError):
        run_experiment(conn, plan(symbol="XYZABC"), SETTINGS, device=CPU, verbose=False)

def test_detailed_outcome_is_consistent_with_the_summary(conn):
    outcome = run_experiment_detailed(conn, plan(flat_cost_bps=0.8), SETTINGS, device=CPU,
                                      verbose=False)
    assert outcome.per_bar.index.equals(outcome.predictions.index)
    assert outcome.per_bar.sum() == pytest.approx(outcome.summary["total_return"])
    assert outcome.summary["run_id"] >= 1
