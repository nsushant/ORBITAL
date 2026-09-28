# Getting started

## Requirements

- Python 3.11 or later.
- PostgreSQL 17 for persistent ingestion. PostgreSQL is optional for parser and
  physics development.
- Docker Desktop only if the optional containerized PostgreSQL route is used.

Run commands from `C:\Users\snigudkar\ORBITAL`, the directory containing the
`SLA` package.

## Python environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r SLA\requirements.txt
```

The core GCAT and LL2 parsers use the Python standard library. Numerical
transfer modules additionally require NumPy, Numba, and h5py. PostgreSQL access
uses psycopg 3.

## Time and unit conventions

- Every versioned input or snapshot has one timezone-aware UTC `as_of` value.
- Every other time, epoch offset, duration, deadline, and time of flight is in
  seconds relative to `as_of` and uses an `_s` suffix.
- Semimajor axes and orbit radii are kilometres.
- Angles are radians in the astrodynamics API.
- `transfer_dv` returns metres per second.
- Mass is kilograms.
- Database source snapshots retain their UTC retrieval timestamp; normalized
  event and schedule times are stored as seconds relative to that timestamp.

## Verify modules without PostgreSQL

```powershell
$env:PYTHONPATH = "C:\Users\snigudkar\ORBITAL"
python -m pytest SLA\dsl\tests SLA\launch_generator\ingestion\tests SLA\storage\tests
```

If pytest is not installed, individual modules can still be imported and the
example DSL files can be validated:

```powershell
python -m SLA.dsl.validate_dsl SLA\dsl\examples\refuelling.json
python -m SLA.dsl.validate_dsl SLA\dsl\examples\repair.json
python -m SLA.dsl.validate_dsl SLA\dsl\examples\deorbit.json
```

## Local PostgreSQL

### Native PostgreSQL

Create a local database and account, then set a connection string in the
current terminal or a local `.env` file that is excluded from source control:

```powershell
$env:SLA_DATABASE_URL = "postgresql://sla:YOUR_PASSWORD@127.0.0.1:5432/sla"
python -m SLA.storage.migrate
```

### Optional Docker route

Copy `SLA\.env.example` to `SLA\.env`, change the password, start Docker
Desktop, and run:

```powershell
cd SLA
docker compose up -d postgres
cd ..
$env:SLA_DATABASE_URL = "postgresql://sla:YOUR_PASSWORD@127.0.0.1:5432/sla"
python -m SLA.storage.migrate
```

The Docker engine must be running. The compose file has been structurally
validated, but a full database integration run has not yet been completed on
this machine because Docker Desktop was stopped.

Local database files for the container route live under `SLA\.data\postgres`
and are excluded from source control.

## Minimal transfer example

```python
from SLA import transfer_dv

dv_mps = transfer_dv(
    a0_km, inclination0_rad, raan0_rad,
    af_km, inclinationf_rad, raanf_rad,
    tof_s,
)
```

A `NaN` result means the analytical search did not find a closing transfer; it
is not a proof that no physical transfer exists.

## Compile an SLA document

```python
from SLA.dsl import load_problem

problem = load_problem("SLA/dsl/examples/refuelling.json")
print(problem.as_of)
print(problem.sla)
print(problem.candidate_servicers())
```

See `SLA/dsl/DSL_V0.1.md` for all fields and semantics.
