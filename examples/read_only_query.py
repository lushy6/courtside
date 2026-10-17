"""Offline demonstration; no writes, credentials, network calls, or model needed."""

from pathlib import Path

from courtside.db import list_tables, open_readonly, run_query
from courtside.errors import QueryError


def main() -> None:
    database = Path(__file__).resolve().parents[1] / "data" / "courtside_nba.db"
    connection = open_readonly(database)
    try:
        tables = list_tables(connection)
        result = run_query(
            connection,
            "SELECT team, count(*) AS lineups FROM lineup_stats "
            "WHERE poss >= 100 GROUP BY team ORDER BY lineups DESC, team LIMIT 5",
            allowed_tables=tables,
        )
        print("Descriptive lineup counts, restricted to at least 100 possessions:")
        print(result.columns)
        for row in result.rows:
            print(row)
        try:
            run_query(connection, "DROP TABLE players", allowed_tables=tables)
        except QueryError as error:
            print(f"Write rejected as expected: {error.kind}")
        else:
            raise RuntimeError("The safety boundary unexpectedly accepted a write")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
