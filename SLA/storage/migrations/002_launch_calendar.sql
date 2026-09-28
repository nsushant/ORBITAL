CREATE TABLE IF NOT EXISTS launch_event (
    source text NOT NULL,
    external_launch_id text NOT NULL,
    source_snapshot_id uuid NOT NULL REFERENCES source_snapshot(snapshot_id) ON DELETE RESTRICT,
    actual_launch_s double precision,
    time_precision text NOT NULL,
    time_uncertain boolean NOT NULL,
    raw_launch_date text NOT NULL,
    vehicle text,
    flight_id text,
    platform text,
    site text,
    pad text,
    ascent_site text,
    ascent_pad text,
    agency text,
    provider_state text,
    launch_code text,
    outcome text NOT NULL CHECK (outcome IN ('success', 'failure', 'unknown', 'other')),
    citation text,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source, external_launch_id)
);

CREATE TABLE IF NOT EXISTS launch_payload (
    source text NOT NULL,
    external_launch_id text NOT NULL,
    payload_id text NOT NULL,
    piece text,
    name text,
    payload_name text,
    owner text,
    owner_state text,
    mass_kg double precision CHECK (mass_kg IS NULL OR mass_kg >= 0),
    dry_mass_kg double precision CHECK (dry_mass_kg IS NULL OR dry_mass_kg >= 0),
    total_mass_kg double precision CHECK (total_mass_kg IS NULL OR total_mass_kg >= 0),
    perigee_km double precision,
    apogee_km double precision,
    inclination_deg double precision,
    orbit_code text,
    orbit_class text,
    program text,
    plane text,
    mission_class text,
    category text,
    payload_result text,
    PRIMARY KEY (source, external_launch_id, payload_id),
    FOREIGN KEY (source, external_launch_id)
        REFERENCES launch_event(source, external_launch_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS announced_launch_state (
    source_snapshot_id uuid NOT NULL REFERENCES source_snapshot(snapshot_id) ON DELETE RESTRICT,
    external_launch_id text NOT NULL,
    name text NOT NULL,
    status text,
    net_s double precision,
    window_start_s double precision,
    window_end_s double precision,
    time_precision text,
    source_last_updated_s double precision,
    provider text,
    vehicle text,
    site text,
    pad text,
    mission_type text,
    orbit_name text,
    orbit_abbrev text,
    PRIMARY KEY (source_snapshot_id, external_launch_id),
    CHECK (window_start_s IS NULL OR window_end_s IS NULL OR window_start_s <= window_end_s)
);

CREATE TABLE IF NOT EXISTS launch_revision (
    revision_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    old_snapshot_id uuid REFERENCES source_snapshot(snapshot_id) ON DELETE RESTRICT,
    new_snapshot_id uuid NOT NULL REFERENCES source_snapshot(snapshot_id) ON DELETE RESTRICT,
    external_launch_id text NOT NULL,
    field_name text NOT NULL,
    old_value jsonb,
    new_value jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT
        (old_snapshot_id, new_snapshot_id, external_launch_id, field_name)
);

CREATE INDEX IF NOT EXISTS launch_event_actual_time_idx
    ON launch_event (actual_launch_s);
CREATE INDEX IF NOT EXISTS launch_event_vehicle_idx
    ON launch_event (vehicle);
CREATE INDEX IF NOT EXISTS launch_payload_program_idx
    ON launch_payload (program);
CREATE INDEX IF NOT EXISTS launch_payload_orbit_idx
    ON launch_payload (orbit_class);
CREATE INDEX IF NOT EXISTS announced_launch_state_launch_idx
    ON announced_launch_state (external_launch_id);
CREATE INDEX IF NOT EXISTS announced_launch_state_net_idx
    ON announced_launch_state (net_s);
CREATE INDEX IF NOT EXISTS launch_revision_launch_idx
    ON launch_revision (external_launch_id, created_at);

INSERT INTO schema_migration(version)
VALUES ('002_launch_calendar')
ON CONFLICT (version) DO NOTHING;
