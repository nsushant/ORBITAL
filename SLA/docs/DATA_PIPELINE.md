# Launch-calendar data pipeline

## Sources

### GCAT

GCAT supplies realized launch history. The importer uses:

- `launchlog.tsv`: launch, provider/agency, vehicle, site, outcome, and payload
  identities.
- `satcat.tsv`: mass and orbital elements for standard catalog objects.
- `psatcat.tsv`: program, constellation plane, class, category, and payload
  result.

GCAT is CC BY 4.0. Preserve the attribution returned by the importer:

```text
Data from GCAT (J. McDowell, planet4589.org/space/gcat)
```

### Launch Library 2

LL2 supplies announced future calendars. Periodic snapshots are required
because the final launch record alone cannot reconstruct what was known before
a delay, cancellation, vehicle change, or site change.

Commercial storage and redistribution terms for LL2 data must be recorded
before customer-facing deployment.

## GCAT importer

```python
from SLA.launch_generator.ingestion import import_gcat

result = import_gcat(
    "launchlog.tsv",
    object_catalog="satcat.tsv",
    payload_catalog="psatcat.tsv",
)

print(len(result.launches))
print(result.attribution)
print(result.launches[0].manifested_payload_mass_kg)
```

The object and payload catalogs are optional. Missing enrichment produces null
fields rather than discarding a launch. Current real-data validation processed
7,414 launches and 31,583 payload rows. Standard `satcat.tsv` enriched 27,852
payloads with mass and 27,851 with an orbit class; auxiliary catalogs will be
added later for the remaining objects.

Dates retain their original text, precision, and uncertainty flag. Exact and
day-precision values are represented in UTC.

## LL2 collector

```python
from SLA.launch_generator.ingestion import LL2Client

snapshot = LL2Client().fetch_upcoming(limit=100, max_pages=20)
print(snapshot.as_of)
print(len(snapshot.launches))
```

The live validation collected all 372 launches reported by the API across its
pages. No validation payload was retained.

Normalized LL2 fields include:

- Stable LL2 launch identifier and name.
- Status.
- NET time and window start/end in seconds from `as_of`.
- Schedule precision and source update time.
- Provider, vehicle, site, and pad.
- Mission type and destination-orbit name/abbreviation.

Raw pages can be captured during collection:

```python
pages = []

def page_sink(url: str, raw: bytes, as_of):
    pages.append((url, raw, as_of))

snapshot = LL2Client().fetch_upcoming(page_sink=page_sink)
```

In production the callback will create one `SourceSnapshot` per raw page. A
collection manifest or combined normalized snapshot must also be stored and
used as the parent of the canonical `announced_launch_state` rows. That final
orchestration is not implemented yet.

## Revision detection

```python
from SLA.launch_generator.ingestion import diff_snapshots

revisions = diff_snapshots(previous_snapshot, current_snapshot)
for revision in revisions:
    print(revision.launch_id, revision.field,
          revision.old_value, revision.new_value)
```

Relative time values are converted back to absolute UTC before comparison, so
changing `as_of` does not create false schedule revisions.

## Repeatable LL2 collection

With PostgreSQL running, collect and compare the current upcoming calendar:

```powershell
$env:SLA_DATABASE_URL = "postgresql://sla:sla-local-dev@127.0.0.1:5432/sla"
python -m SLA.launch_generator.collect_ll2
```

The default network policy allows 60 seconds per page and retries transient
timeouts, connection failures, rate limits, and common server errors three
times. Override it with `--timeout-s` and `--retries` when needed.

Each run archives every raw LL2 page, creates a collection manifest, stores the
normalized announced states, and compares them with the preceding completed
collection. Added, removed, rescheduled, and otherwise changed launch fields are
written to `launch_revision`. The command prints snapshot identifiers, page and
launch counts, revision counts by field, and logical raw-storage usage.

An interrupted paginated request can leave archived raw pages, but it does not
create a completed manifest or canonical calendar. The next successful run uses
the latest manifest that has canonical rows as its comparison baseline.

## Source snapshot repository

```python
import psycopg
from SLA.storage import SnapshotRepository, SourceSnapshot

with psycopg.connect(database_url) as connection:
    repository = SnapshotRepository(connection)
    stored = repository.add(SourceSnapshot(
        source="ll2",
        retrieved_at=as_of,
        request_uri=url,
        content=raw_bytes,
        media_type="application/json",
        source_version="2.3.0",
        licence_status="unresolved",
    ))
```

Identical bytes reuse the same `source_object`; each retrieval still creates a
separate `source_snapshot`.

## Canonical calendar repository

```python
from SLA.storage import LaunchCalendarRepository

repository = LaunchCalendarRepository(connection)
launch_count, payload_count = repository.upsert_gcat(
    gcat_result,
    as_of=gcat_retrieval_time,
    snapshot_id=gcat_source_snapshot_id,
)

repository.add_ll2_snapshot(ll2_snapshot, ll2_collection_snapshot_id)
repository.add_revisions(
    revisions,
    old_snapshot_id=previous_collection_snapshot_id,
    new_snapshot_id=ll2_collection_snapshot_id,
)
```

## Database migrations

- `001_source_snapshots.sql`: content-addressed raw objects and retrievals.
- `002_launch_calendar.sql`: realized events, payloads, announced states, and
  revisions.

Apply them with:

```powershell
python -m SLA.storage.migrate
```

Migrations are ordered and idempotent. They have been applied and the snapshot
repository has passed a live write, read, deduplication, and cleanup test against
the local PostgreSQL container.

## Data-quality rules

- Preserve source identifiers and original text.
- Never fill missing values with zero.
- Keep observed, derived, predicted, and assumed values distinguishable.
- Never use information published after a training row's `as_of` cutoff.
- Do not describe technical residual mass as commercially bookable capacity.
- Archive every forecast used for an SLA decision.
- Preserve model, calibration, evidence, and threshold versions.

## Current limitations

- GCAT enrichment currently accepts one object and one payload catalog at a
  time; auxiliary catalogs are not yet merged automatically.
- LL2 collection-manifest persistence is not yet orchestrated.
- LL2 history currently begins with the first stored snapshot from 2026-09-20;
  revision-based training depth will grow as periodic collection runs.
- Vehicle technical-performance tables are not yet implemented.
- The Bayesian predictor and risk engine are not yet implemented.


## Leakage-safe feature builder

`SLA.launch_generator.features` converts normalized GCAT history and timestamped
LL2 snapshots into four typed training datasets:

- **count rows**: provider launch counts over a chosen future horizon;
- **mission rows**: delay and realized-launch labels for announced missions;
- **mark rows**: vehicle, site, and destination-orbit outcomes;
- **capacity rows**: manifested payload count and observed payload mass.

```python
from datetime import datetime, timezone
from SLA.launch_generator import build_feature_bundle, chronological_split, write_jsonl

bundle = build_feature_bundle(
    gcat_result,
    ll2_snapshots,
    horizons_s=[30 * 86400, 90 * 86400],
    provider_aliases={"spx": "spacex", "SpaceX": "spacex"},
    launch_crosswalk={"ll2-launch-uuid": "GCAT-LAUNCH-ID"},
    label_cutoff=datetime(2026, 1, 1, tzinfo=timezone.utc),
)
write_jsonl("training/counts.jsonl", bundle.counts)
train, calibration, test = chronological_split(
    bundle.counts,
    train_before=datetime(2024, 1, 1, tzinfo=timezone.utc),
    calibration_before=datetime(2025, 1, 1, tzinfo=timezone.utc),
)
```

All durations are seconds and every row has a timezone-aware UTC `as_of`.
Features only use records available by `as_of`. Count rows whose complete label
horizon extends beyond `label_cutoff` are omitted. Mission rows after the actual
launch are omitted. Splits are chronological rather than random.

Provider aliases and LL2-to-GCAT launch identifiers must be supplied explicitly.
The builder does not silently fuzzy-match entities. Unmatched mission rows remain
unlabelled, so missing crosswalk entries are not interpreted as cancellations.

## Current local database baseline

The first live import was completed on 2026-09-20 using GCAT data updated
2026-09-18 and a current LL2 upcoming-calendar snapshot:

- 7,414 GCAT launch events;
- 31,583 GCAT payload records;
- 371 LL2 announced launch states across four archived API pages;
- nine provenance snapshots and collection manifests;
- approximately 25 MB total PostgreSQL database size.

The three GCAT inputs and every LL2 response page are stored as immutable,
content-addressed source objects. Canonical rows reference collection manifests.
