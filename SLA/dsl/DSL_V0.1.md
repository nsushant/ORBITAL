# OOS SLA DSL v0.1

## Purpose

The DSL is the versioned boundary between contract descriptions, infrastructure state, launch-access forecasts, and the planning engine. JSON is the canonical representation. Algorithms and equations remain in code and are referenced by model identifier.

## Documents

Version 0.1 uses one problem bundle containing:

1. `sla`: promised service, client, success rule, reliability target, and commercial remedies.
2. `provider_state`: servicers, depots, inventory, and existing commitments at the information cutoff.
3. `launch_access`: configuration of the joint generative launch-calendar model.
4. `decision_policy`: decision stages and freeze rules.
5. `models`: versioned transfer, planning, and service plugins.

The schema is `schema/sla-dsl-v0.1.schema.json`. Complete refuelling, repair, and deorbit examples are under `examples/`.

## Python interface

```python
from SLA.dsl import load_problem

problem = load_problem("SLA/dsl/examples/refuelling.json")

print(problem.sla.service)
print(problem.sla.reliability.minimum)
print(problem.candidate_servicers())
```

`load_problem` performs JSON Schema and semantic validation before returning an
immutable `SLAProblem`. Pass `validate=False` only for trusted internal data or
low-level tests. Invalid documents raise `DSLCompileError` containing all
detected validation errors.

## Semantics

- `as_of` is the single absolute time and uses an accurate ISO 8601 UTC timestamp ending in `Z`.
- Every other instant is stored as seconds relative to `as_of`; negative values denote times before it.
- Every duration is stored directly in seconds. Field names carrying time end in `_s`.
- Orbital angles are radians and semimajor axes are kilometres.
- Money always carries an ISO 4217 currency code.
- Quantities always carry units.
- `as_of` is the information cutoff and time origin: later information must not enter a historical solve.
- Reliability is evaluated over scenarios emitted by the referenced launch-access model.
- Decisions assigned to an earlier stage must remain identical across scenarios that are indistinguishable at that stage.
- `NaN` from the analytical transfer oracle means `ANALYTICAL_TRANSFER_NOT_FOUND`, not proven physical infeasibility.

## Controlled rule language

Success conditions use comparisons joined by `all`, `any`, and `not`. Arbitrary executable expressions are excluded. New metrics require a schema and evaluator version change.

## Extension policy

Version 0.1 supports `commodity_delivery`, `repair`, and `deorbit`. Add a new service through a typed schema branch and a registered service plugin. Do not place solver equations, Python expressions, or unversioned policy logic in JSON.

## Validation layers

1. JSON parsing.
2. JSON Schema validation.
3. Semantic checks that JSON Schema cannot express conveniently, including time ordering, model/service compatibility, referenced decision names, capability availability, consistent units, inventory sufficiency, and information-cutoff rules.
4. Model-domain checks, such as applicability of the Edelbaum/J2 oracle.

Version 0.1 is frozen only after all three examples pass schema and semantic validation and the refuelling example executes end to end.

## Time conversion

The compiler retains `as_of` as a timezone-aware Python `datetime` and all
relative values as seconds. Calendar time is recovered only at an interface:

```python
from datetime import timedelta

calendar_time = problem.as_of + timedelta(seconds=problem.sla.service_window.latest_s)
```

Physics and optimization code must consume the `_s` values directly. It must
not divide by 86,400 internally or infer units from unqualified numeric fields.
