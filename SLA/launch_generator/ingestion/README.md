# GCAT importer

The importer reads GCAT's tab-separated derived orbital launch log and can
enrich it from the object and payload catalogs. It uses only the Python standard
library and does not require PostgreSQL.

```python
from SLA.launch_generator.ingestion.gcat import import_gcat

result = import_gcat(
    "launchlog.tsv",
    object_catalog="satcat.tsv",
    payload_catalog="psatcat.tsv",
)
```

The output groups payload rows by GCAT launch tag and preserves GCAT identifiers,
raw dates, date precision, evidence citation, mass fields, orbital elements,
program, plane, and mission classification. Mass and orbit fields remain null
when enrichment files are omitted.

GCAT is CC BY 4.0. Preserve `result.attribution` in datasets and product exports.

## Launch Library 2 collector

`LL2Client.fetch_upcoming()` follows API pagination and returns one normalized
`LL2Snapshot`. Its `as_of` value is an exact UTC timestamp; advertised launch
times and update times are stored as seconds relative to `as_of`.

Pass a `page_sink(url, raw_bytes, as_of)` callback to persist every raw API page
through the PostgreSQL `SnapshotRepository`. `diff_snapshots(old, new)` reports
added, removed, and field-level revisions while comparing relative time fields
as absolute UTC instants.

The collector validates pagination hosts, limits page counts, retries HTTP 429
responses, and sends a named user agent. Commercial storage and redistribution
of LL2 data remain subject to written source terms.
