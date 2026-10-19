"""Central settings: paths, symbols, timeframes, DB location, training defaults."""
from pathlib import Path

# config.py -> fxml -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DB_PATH = DATA_DIR / "forex.db"

# What we download and train on (start small, widen later)
SYMBOLS = ["EURUSD"]
TIMEFRAMES = ["H1"]
HISTORY_START = "1999-01-01"  # UTC; real EURUSD data begins here
TRAIN_START = "2015-01-01"  # UTC; earlier data is stored but its timestamps are less reliable
MAX_BARS = 5_000_000  # upper bound per request; larger than any history we have
LABEL_HORIZON = 4  # predict the move over the next N bars
LOOKBACK = 48  # bars of history per sample (2 days of H1)
HIDDEN_SIZE = 64
DROPOUT = 0.2
POINT_SIZE = 0.00001  # EURUSD quotes 5 digits, so 1 point = 0.00001
VOL_HORIZON = 24  # forecast realised volatility over the next N bars (one trading day of H1)
