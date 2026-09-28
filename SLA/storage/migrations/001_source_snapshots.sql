CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS schema_migration (
    version text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS source_object (
    object_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash text NOT NULL UNIQUE CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    media_type text NOT NULL,
    content bytea NOT NULL,
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (octet_length(content) = size_bytes)
);

CREATE TABLE IF NOT EXISTS source_snapshot (
    snapshot_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL CHECK (length(source) > 0),
    retrieved_at timestamptz NOT NULL,
    request_uri text NOT NULL CHECK (length(request_uri) > 0),
    object_id uuid NOT NULL REFERENCES source_object(object_id) ON DELETE RESTRICT,
    source_version text,
    licence_status text NOT NULL DEFAULT 'unresolved'
        CHECK (licence_status IN ('cleared', 'internal_only', 'unresolved', 'prohibited')),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS source_snapshot_source_retrieved_idx
    ON source_snapshot (source, retrieved_at DESC);

CREATE INDEX IF NOT EXISTS source_snapshot_object_idx
    ON source_snapshot (object_id);

INSERT INTO schema_migration(version)
VALUES ('001_source_snapshots')
ON CONFLICT (version) DO NOTHING;
