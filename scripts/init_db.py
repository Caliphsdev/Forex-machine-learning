"""CLI: create the SQLite database and tables."""
from fxml import config
from fxml.db.connection import connect
from fxml.db.schema import init_db


def main() -> None:
    conn = connect()
    init_db(conn)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    print(f"Database: {config.DB_PATH}")
    print("Tables:", [t[0] for t in tables])
    conn.close()


if __name__ == "__main__":
    main()
