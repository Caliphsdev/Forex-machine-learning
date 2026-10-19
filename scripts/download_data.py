"""CLI: download MT5 history into SQLite."""
from fxml.db.connection import connect
from fxml.db.schema import init_db
from fxml.ingest.downloader import download_all


def main() -> None:
    conn = connect()
    init_db(conn)
    download_all(conn)
    conn.close()


if __name__ == "__main__":
    main()
