# Local PostgreSQL storage

This layer stores every source retrieval while deduplicating identical response
bodies by SHA-256. The database is bound to `127.0.0.1` and is intended for the
local MVP.

## Start

1. Copy `.env.example` to `.env` and replace the password.
2. Run `docker compose up -d postgres` from the `SLA` directory.
3. Install the Python dependency with `pip install -r requirements.txt`.

PostgreSQL applies `storage/migrations/001_source_snapshots.sql` automatically
when it creates a new empty data directory. Apply all migrations to either a
native or containerized database with:

```powershell
python -m SLA.storage.migrate
```

The command reads `SLA_DATABASE_URL` and safely reruns the ordered idempotent
SQL files.

Local database files live under `SLA/.data/postgres` and must not be committed.
Back up the database before deleting that directory.
