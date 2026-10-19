"""Open/close SQLite connections, apply pragmas."""
import sqlite3
from pathlib import Path

from fxml import config


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Connect to the SQLite database, applying necessary pragmas."""
    path = Path(db_path) if db_path else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
