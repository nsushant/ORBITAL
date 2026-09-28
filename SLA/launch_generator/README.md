# Launch calendar generator

This package owns the data and modelling boundary for uncertain future launch access.
It converts public launch records and timestamped schedule revisions into canonical
objects and leakage-safe training rows that can support a joint probabilistic launch
calendar generator.

## Current contents

- `ingestion/`: GCAT history and Launch Library 2 schedule ingestion.
- `models.py`: versioned evidence, launch-opportunity, scenario and realization objects.
- `features.py`: chronological training rows for launch count, mission marks and capacity.
- `archive.py`: forecast snapshot serialization.
- `scoring.py`: forecast calibration metrics.
- `collect_ll2.py`: LL2 snapshot collector.
- `IMPLEMENTATION_PLAN.md`: planned Bayesian calendar and capacity model.

## Package interface

Import through `SLA.launch_generator`. Times inside model objects are seconds relative
to a timezone-aware UTC `as_of` timestamp.

```python
from SLA.launch_generator import build_feature_bundle, chronological_split
from SLA.launch_generator.ingestion import LL2Client, import_gcat
```

The generated output is a weighted collection of joint launch scenarios. Each launch
opportunity keeps the coupled time, provider, vehicle, site, destination orbit,
purchasable capacity, price, booking deadline, delay and cancellation information.
The SLA planner consumes these scenarios; it does not own the forecasting model.

Database persistence remains in `SLA.storage` because it is shared infrastructure.
`SLA.storage.calendar_repository` is the adapter between this package and PostgreSQL.

## Validation

From the repository root:

```text
python -m pytest SLA/launch_generator SLA/demand_forecast SLA/storage/tests
```
