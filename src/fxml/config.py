"""Central settings: paths, symbols, timeframes, DB location, training defaults."""
from pathlib import Path

# config.py -> fxml -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DB_PATH = DATA_DIR / "forex.db"


# Price value of one point per symbol, from the broker's `digits` (5 digits -> 0.00001, 3 -> 0.001)
POINT_SIZES = {
    "EURUSD": 0.00001, "GBPUSD": 0.00001, "AUDUSD": 0.00001, "USDCAD": 0.00001,
    "USDCHF": 0.00001, "NZDUSD": 0.00001, "EURGBP": 0.00001,
    "USDJPY": 0.001, "EURJPY": 0.001, "GBPJPY": 0.001, "AUDJPY": 0.001,
}
POINT_SIZE = POINT_SIZES["EURUSD"]  # kept for the older notebooks
SYMBOLS = list(POINT_SIZES)  # everything we download


TIMEFRAMES = ["H1"]
HISTORY_START = "1999-01-01"  # UTC; real EURUSD data begins here
TRAIN_START = "2015-01-01"  # UTC; earlier data is stored but its timestamps are less reliable
MAX_BARS = 5_000_000  # upper bound per request; larger than any history we have
LABEL_HORIZON = 4  # predict the move over the next N bars
LOOKBACK = 48  # bars of history per sample (2 days of H1)
HIDDEN_SIZE = 64
DROPOUT = 0.2
VOL_HORIZON = 24  # forecast realised volatility over the next N bars (one trading day of H1)
