# SLA demonstrator documentation

This package implements the early foundation of an on-orbit-servicing SLA
decision product. It currently contains analytical transfer estimates, an SLA
domain-specific language, service plugins, launch-access forecast contracts,
GCAT and Launch Library 2 ingestion, and a local PostgreSQL persistence layer.

## Documents

- [`SLA_TOOL_TECHNICAL_DESCRIPTION.tex`](SLA_TOOL_TECHNICAL_DESCRIPTION.tex):
  complete system description, mathematical formulation, solver architecture,
  decision logic, validation boundary, and current limitations.
- [`GETTING_STARTED.md`](GETTING_STARTED.md): installation, configuration, and
  verification.
- [`ARCHITECTURE.md`](ARCHITECTURE.md): module boundaries and end-to-end data
  flow.
- [`DATA_PIPELINE.md`](DATA_PIPELINE.md): GCAT, LL2, snapshots, revisions, and
  PostgreSQL usage.
- [`../dsl/DSL_V0.1.md`](../dsl/DSL_V0.1.md): SLA input language.
- [`../launch_generator/IMPLEMENTATION_PLAN.md`](../launch_generator/IMPLEMENTATION_PLAN.md):
  remaining Bayesian forecast and risk implementation plan.
- [`../demand_forecast/README.md`](../demand_forecast/README.md): Bayesian
  service-demand model, posterior update, and scheduler interface.
- [`../mvp/README.md`](../mvp/README.md): localhost portfolio planner,
  example input, model boundary, and run instructions.
- [`../research/sla_portfolio_mvp_demo.ipynb`](../research/sla_portfolio_mvp_demo.ipynb):
  executable decision walkthrough and launch-risk sensitivity study.
- [`../oos_product_research_outline.md`](../oos_product_research_outline.md):
  product and research rationale.
- [`raan_closure_defect.md`](raan_closure_defect.md) and
  [`lu_normalisation.md`](lu_normalisation.md): analytical transfer notes.

## Current status

| Component | Status |
|---|---|
| Edelbaum/J2 transfer estimate | Implemented and regression tested |
| SLA DSL schema/compiler | Implemented with three examples |
| Commodity-delivery service plugin | Implemented |
| Forecast/scenario data contracts | Implemented |
| GCAT launch importer | Implemented and tested on current data |
| LL2 upcoming-launch collector | Implemented and live-API tested |
| Snapshot/revision database schema | Implemented |
| PostgreSQL repositories | Implemented; live DB integration pending |
| Bayesian service-demand predictor | Implemented and tested |
| Joint launch/deployment/demand scenarios | Implemented |
| Bayesian launch/capacity predictor | Designed, not implemented |
| Residual calibration | Designed, not implemented |
| SLA scenario reliability and economic evaluator | Implemented in the research MVP |
| Local SLA portfolio demonstrator | Implemented and tested |
| Production user interface | Not implemented |
