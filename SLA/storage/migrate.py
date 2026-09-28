"""Apply ordered, idempotent PostgreSQL migrations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def apply_migrations(connection: Any, directory: str | Path | None = None) -> list[str]:
    root = Path(directory) if directory else Path(__file__).with_name("migrations")
    migrations = sorted(root.glob("[0-9][0-9][0-9]_*.sql"))
    if not migrations:
        raise FileNotFoundError(f"no migrations found in {root}")
    applied: list[str] = []
    for migration in migrations:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute(migration.read_text(encoding="utf-8"))
        applied.append(migration.name)
    return applied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply SLA PostgreSQL migrations")
    parser.add_argument("--database-url", default=os.environ.get("SLA_DATABASE_URL"))
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("set SLA_DATABASE_URL or pass --database-url")
    try:
        import psycopg
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before running migrations") from exc
    with psycopg.connect(args.database_url) as connection:
        for name in apply_migrations(connection, args.directory):
            print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
