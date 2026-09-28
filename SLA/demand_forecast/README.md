# Bayesian service-demand MVP

This package generates future SLA requests conditional on the observable
satellite population and sampled launch calendars.

## Model

- Confirmed orders appear in every scenario.
- Pipeline orders convert with explicit probabilities.
- Additional demand follows a mixed Poisson process.
- Each service type has a Gamma posterior over its event rate.
- Satellite age, design life and health scale exposure to demand.
- Each event carries optimizer-relevant marks: target orbit, deadline, service
  duration, value, inventory requirement and priority.
- Planned satellites become demand-generating assets only in launch scenarios
  containing their linked launch opportunity.

The rate unit is events per **weighted satellite-second**. All other times are
seconds relative to the exact UTC `as_of` timestamp in the forecast snapshot.

## Learning

For a service type with prior `Gamma(shape, rate_s)`, observing `n` requests
during `E` weighted satellite-seconds produces:

```text
shape'  = shape + n
rate_s' = rate_s + E
```

This is the first interpretable research baseline. It can later be replaced by
a richer regression or point-process model without changing the scenario
objects or scheduler adapter.

## Link to SLA scheduling

`to_scheduler_demands` converts a `DemandScenario` into the existing
`SLA.schedule.Demands` arrays using the target-to-cost-table node mapping.
`joint_forecast` pairs every launch realization with one or more conditional
demand draws. The SLA acceptance evaluator can run the scheduler on each joint
scenario and use the supplied probabilities to calculate reliability.
