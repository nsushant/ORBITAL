# SLA demonstrator support package

This package contains the physics, data, and domain foundations for an on-orbit-servicing SLA acceptance and risk product.

## Documentation

Start with [`docs/INDEX.md`](docs/INDEX.md), then use:

- [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) for installation and verification.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for modules and data flow.
- [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) for GCAT, Launch Library 2, PostgreSQL, and revision tracking.
- [`dsl/DSL_V0.1.md`](dsl/DSL_V0.1.md) for the SLA JSON language.
- [`launch_generator/IMPLEMENTATION_PLAN.md`](launch_generator/IMPLEMENTATION_PLAN.md) for the remaining forecast and risk build.

## Implemented components

- Conservative thrust-coast-thrust Edelbaum transfer estimate with J2 RAAN closure.
- SLA DSL v0.1 with refuelling, repair, and deorbit examples.
- Typed launch opportunities, scenarios, forecast snapshots, outcomes, and probabilistic scores.
- GCAT and Launch Library 2 collection with a canonical PostgreSQL launch schema.
- Bayesian state-dependent service-demand scenarios with posterior updates.
- Coupled launch, deployment and demand planning scenarios.
- Localhost SLA portfolio demonstrator with a binary manifest MILP, MDLS routing, Edelbaum/J2 transfer table, inventory balances and scenario reliability.

## Current development boundary

The end-to-end research MVP can construct a manifest from individual payload candidates, enforce launch capacity, route servicers under uncertainty and make an SLA acceptance decision. The Bayesian launch/capacity predictor remains the main data-model extension. Larger candidate sets should replace exact enumeration with a production MILP backend.

## Local portfolio demonstrator

Run from `C:\Users\snigudkar\ORBITAL`:

```powershell
python -m SLA.mvp.server
```

Then open `http://127.0.0.1:8765`. You can also double-click `SLA\start_mvp.cmd`. The planner reports the manifest decisions, MDLS routes, SLA reliability, launch calendar, resupply calendar and commercial result.

## Time and unit convention

- One absolute, timezone-aware UTC `as_of` timestamp per versioned input.
- All other times and durations are seconds relative to `as_of`.
- Semimajor axes are kilometres, angles are radians, masses are kilograms, and transfer delta-v is metres per second.

## Core transfer use

```python
from SLA import transfer_dv

dv_mps = transfer_dv(
    a0_km, inclination0_rad, raan0_rad,
    af_km, inclinationf_rad, raanf_rad,
    tof_s,
)
```

A `NaN` result means the analytical search did not find a closing transfer; it does not prove physical infeasibility. See the technical notes under `docs/`.
