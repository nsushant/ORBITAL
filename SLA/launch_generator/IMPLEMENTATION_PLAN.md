# Launch-access and SLA risk MVP implementation plan

## Fixed MVP boundary

- Data sources: GCAT releases and immutable Launch Library 2 snapshots.
- Forecast marks: time, provider, vehicle, site, broad orbit class, manifested
  mass, and projected accessible capacity.
- One orbital-service demonstrator consumes generated scenario calendars.
- No automated document/LLM extraction in the first implementation.
- No claim that inferred residual capacity is observed bookable capacity.

## Build order

1. **Raw snapshot store**
   - Save source, retrieval UTC, request, response bytes, content hash, source
     version, and licence status.
   - Never overwrite a snapshot.
2. **Canonical records**
   - Normalize GCAT realised events and LL2 announced missions.
   - Maintain stable internal identifiers and explicit source-record mappings.
   - Derive revision events by comparing consecutive LL2 snapshots.
3. **Feature snapshots**
   - Construct each training row using only information available at its
     `as_of` timestamp.
   - Include recent cadence, manifest count, provider, vehicle, site, season,
     lead time, revision history, orbit class, payload/operator information,
     and optional versioned market signals.
4. **Bayesian predictor**
   - Hierarchical negative-binomial background counts.
   - Discrete-time announced-mission launch/delay/cancellation hazards.
   - Conditional categorical site, vehicle, and orbit marks.
   - Conditional manifested-mass distribution.
   - Accessibility-factor priors by mission/evidence class.
5. **Scenario generator**
   - Sample announced and background launches jointly.
   - Derive technical residual and projected accessible capacity.
   - Emit existing `ForecastSnapshot`, `LaunchScenario`, and
     `LaunchOpportunity` objects.
6. **Calibration and scoring**
   - Rolling-origin backtests.
   - Brier score, count log score/error, timing coverage/CRPS, mark log score,
     capacity coverage, and calibration plots.
   - Residual calibration fitted only on the calibration period.
7. **SLA risk evaluator**
   - Evaluate DSL success in every scenario.
   - Return reliability, expected loss, tail loss, rating, evidence confidence,
     main drivers, incremental portfolio risk, and mitigation comparison.
8. **Outcome feedback**
   - Match realised launches and SLA outcomes to archived forecasts/decisions.
   - Update Bayesian posteriors on a controlled release cadence.
   - Promote a model only after chronological tests improve calibration and
     decision quality.

## Minimum tables

### `source_snapshot`

`snapshot_id`, `source`, `retrieved_at_utc`, `request_uri`, `content_hash`,
`raw_object_uri`, `source_version`, `licence_status`.

### `launch_event`

`launch_id`, actual time, provider, vehicle, site, outcome, orbit class,
payload count/mass, GCAT record identifiers, provenance flags.

### `announced_launch_state`

`as_of_utc`, `launch_id`, announced time/window, precision, provider, vehicle,
site, status, orbit class, LL2 record identifier, snapshot identifier.

### `launch_revision`

`revision_id`, `launch_id`, `observed_at_utc`, changed field, old value, new
value, source snapshot.

### `vehicle_performance`

Vehicle/version, orbit class, technical capacity kg, assumptions, source,
effective dates, evidence grade.

### `forecast_run`

Forecast identifier, `as_of_utc`, horizon seconds, model/calibration versions,
feature snapshot, seed, scenario archive, scores when realised.

### `sla_decision_run`

Decision identifier, SLA and portfolio snapshots, forecast identifier, policy,
reliability, expected/tail loss, risk label, evidence confidence, drivers,
mitigations, and selected decision.

### `sla_outcome`

Decision/SLA identifier, fulfilment, completion time, failure cause, realised
cost/penalty/claim, mitigation used, portfolio effects, observation quality.

## First model comparison

- Provider historical-rate baseline.
- Current announced manifest without delay correction.
- Bayesian cadence model without market signals.
- Bayesian cadence plus announced-mission hazard model.
- Add each candidate market signal separately.

The selected model must improve chronological probabilistic scores and not
degrade SLA decision quality. Complexity is not a selection criterion.

## Initial risk output contract

For each SLA return:

- Fulfilment probability and calibration interval.
- Late-completion and complete-failure probabilities.
- Expected loss and 95th-percentile loss.
- Low/Moderate/High/Severe label with versioned thresholds.
- Evidence-confidence grade.
- Three leading scenario-derived risk drivers.
- Incremental portfolio risk.
- Reliability and loss before and after each candidate mitigation.

## Implementation acceptance tests

- Re-running a forecast from its archived inputs and seed reproduces scenarios.
- No feature row contains information published after its `as_of` timestamp.
- Scenario probabilities and calendar records pass schema/unit validation.
- Impossible provider--vehicle--site combinations cannot be sampled.
- Accessible capacity never exceeds technical capacity and is always labelled
  inferred unless directly offered by a cited source.
- Backtests compare forecasts with later GCAT outcomes without leakage.
- Risk labels can be reconstructed from quantitative outputs and threshold
  version.
- Model promotion requires a saved comparison report against all baselines.

## Product and ownership constraints

- Keep collectors, schemas, Bayesian model, calibration, and risk logic in our
  codebase.
- Record the licence and attribution requirement for every source and software
  dependency.
- GCAT is CC BY and requires citation.
- Treat LL2 commercial storage, derived use, and redistribution as unresolved
  until written terms are recorded.
- Use a provider-neutral extraction interface if document parsing is added
  later; no LLM output writes directly to the canonical calendar.
