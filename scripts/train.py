"""CLI: run a walk-forward direction experiment and log it to SQLite."""
import argparse

from fxml import config
from fxml.db.connection import connect
from fxml.db.schema import init_db
from fxml.training.experiment import Settings
from fxml.training.pipeline import ExperimentPlan, run_experiment


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and log a walk-forward experiment.")
    parser.add_argument("--name", required=True, help="label stored with the run")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--start", default=config.TRAIN_START, help="first bar, UTC date")
    parser.add_argument("--end", default=None, help="exclusive end date; default = latest bar")
    parser.add_argument("--horizon", type=int, default=config.LABEL_HORIZON,
                        help="bars ahead to predict")
    parser.add_argument("--train-size", type=int, default=31_000)
    parser.add_argument("--test-size", type=int, default=3_100)
    parser.add_argument("--flat-cost-bps", type=float, default=None,
                        help="charge this flat cost per trade instead of the recorded spread")
    parser.add_argument("--notes", default="")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    conn = connect()
    init_db(conn)
    plan = ExperimentPlan(
        name=args.name, symbol=args.symbol, timeframe=args.timeframe,
        start=args.start, end=args.end, train_size=args.train_size,
        test_size=args.test_size, flat_cost_bps=args.flat_cost_bps, notes=args.notes,
    )
    summary = run_experiment(conn, plan, Settings(horizon=args.horizon))
    conn.close()

    print(f"\nrun {summary['run_id']} saved as {args.name!r}")
    for key, value in summary.items():
        if key != "run_id":
            print(f"  {key:<28} {value:.4f}" if isinstance(value, float) else f"  {key:<28} {value}")


if __name__ == "__main__":
    main()
