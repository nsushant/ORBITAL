# OOS Service Feasibility and Commitment Planner

## 1. Product and research objective

Build a decision-support system for an on-orbit servicing (OOS) provider that answers:

> Given the provider's current assets, inventory, accepted SLAs, and uncertain future launch opportunities, should a new SLA be accepted, at what price and reliability level, and which decisions must be committed now to deliver it?

The launch-market model is an input to this decision. The principal contribution is the conversion of customer promises into infrastructure deployment, inventory, fleet, launch-reservation, and servicing decisions.

### Primary users

- OOS commercial and mission-planning teams.
- OOS business-development teams preparing bids and SLA terms.
- Orbital logistics providers expanding into servicing.
- Satellite operators comparing service offers with replacement or disposal.
- Agencies, insurers, and investors evaluating the credibility of an OOS service plan.

### Decisions supported

- Accept, reject, defer, or renegotiate an SLA request.
- Quote a base price, reliability tier, and risk premium.
- Select depot locations and deployment dates.
- Deploy and allocate servicers.
- Set initial stock and resupply plans.
- Reserve primary and backup launch opportunities.
- Decide which assignments remain provisional and which become frozen.
- Replan after a launch, demand, inventory, or vehicle-state update.

## 2. Scope boundaries

### Inside the product

- OOS portfolio acceptance and feasibility.
- Launch-access scenarios as viewed by an OOS customer.
- Deployment, inventory, resupply, vehicle allocation, and servicing routes.
- SLA reliability, penalties, and commitment timing.
- Rolling-horizon replanning and recovery.

### Outside the first product

- The launch provider's complete internal manifest optimizer.
- Detailed launch-vehicle trajectory design.
- High-fidelity spacecraft guidance and control.
- Full spacecraft engineering and certification.
- Predictions of confidential capacity presented as observed facts.

The product may use simplified compatibility and launch-availability models while exposing their provenance and uncertainty.

Detailed research on a future learned trajectory-optimization layer has been deferred to
[`trajectory_optimization_research_note.md`](trajectory_optimization_research_note.md). The first demonstrator uses the existing analytical transfer model and records its validity limits and conservative bias.

The concrete build sequence, data contracts, model comparisons, and acceptance
tests for the launch-access and risk MVP are maintained in
[`launch_generator/IMPLEMENTATION_PLAN.md`](launch_generator/IMPLEMENTATION_PLAN.md).

## 3. Conceptual system

The system contains three linked models:

1. **Launch-access model:** produces possible calendars visible to the OOS provider, including time, insertion orbit, purchasable capacity, price, and commitment deadlines.
2. **OOS logistics model:** maps each calendar to feasible infrastructure deployment, stocking, resupply, vehicle movement, and service routes.
3. **SLA decision model:** evaluates acceptance, price, reliability, penalties, and the decisions that must be committed at the current epoch.

At decision epoch \(\tau\), the state is:

\[
s_\tau=(\text{accepted SLAs},\text{frozen commitments},\text{deployed assets},
\text{vehicle states},\text{inventory},\text{current launch offers}).
\]

The system returns a policy rather than only a fixed plan:

\[
\pi_\tau: (s_\tau,\text{information revealed}) \mapsto
(\text{commitments and adaptive actions}).
\]

### 3.1 Joint generative launch-access model

The launch-access layer is formulated as a conditional generative model rather
than a collection of independent forecasts. At planning epoch \(\tau\), it
samples complete future calendars from

\[
\mathcal C_\tau^{(s)}\sim
p\!\left(\mathcal C_\tau\mid I_\tau\right),
\]

where \(I_\tau\) is the public information available at that date. It may
contain historical launches, the currently announced manifest, dated schedule
revisions, vehicle activity, launch-site activity, target-orbit patterns,
government contracts, constellation deployment plans, vehicle performance,
published rideshare offers, and timestamped provider or news announcements.

Each generated launch is one coupled record:

\[
\ell=(t,s,v,o,c,p,b,q),
\]

with launch time/window \(t\), site \(s\), provider and vehicle \(v\), target
orbit \(o\), projected purchasable capacity \(c\), price \(p\), booking or
integration deadline \(b\), and evidence/confidence class \(q\). Date, site,
vehicle, orbit, capacity, and price must not be sampled independently.

One possible factorisation is

\[
\begin{aligned}
p(\mathcal C\mid I)={}&p(N,\{t_l\}\mid I)
\prod_l p(v_l,s_l\mid t_l,I)\\
&\times p(o_l\mid v_l,s_l,t_l,I)
p(c_l\mid o_l,v_l,s_l,t_l,I)\\
&\times p(p_l,b_l\mid c_l,o_l,v_l,I).
\end{aligned}
\]

A marked temporal point process is a natural eventual implementation: launch
times are events, and site, vehicle, orbit, capacity, price, and evidence are
their marks. Simpler conditional count and mark models should be retained as
interpretable baselines.

Capacity is treated as a partially observed market variable rather than as the
difference between headline vehicle performance and manifested payload mass:

\[
C_{\mathrm{technical}}
\rightarrow C_{\mathrm{allocated}}
\rightarrow C_{\mathrm{remaining}}
\rightarrow C_{\mathrm{purchasable}}.
\]

Vehicle guides and government performance documents inform technical
capacity. Payload manifests, satellite masses, constellation plans, contracts,
and operator-provider relationships inform allocated demand. Mission type,
rideshare announcements, integration lead time, and provider practice inform
whether any physical residual is commercially accessible. Until operator or
broker data are available, the output must be labelled **projected purchasable
capacity under stated assumptions**.

Every opportunity carries an evidence class:

- **A:** a publicly offered slot or stated available capacity.
- **B:** an announced rideshare opportunity with incomplete availability.
- **C:** a known mission with an inferred technical residual and uncertain
  commercial access.
- **D:** a statistically generated future opportunity.

The public-information extension continuously converts provider releases,
government reports, contract announcements, constellation plans, licensing or
range information, and relevant news into timestamped structured assertions.
Each assertion preserves its source, publication date, extracted value,
confidence, and status as observed, announced, or inferred. Initially, machine
extraction should be followed by human verification.

The generator produces scenario calendars for the SLA optimiser. Public data
provide a useful prior; later operator, broker, or customer-specific data can
condition the same model on actual slots, reservation status, integration
deadlines, negotiated prices, and cancellation terms.

### 3.2 Continual forecast improvement

The launch-access product operates as a forecast--observe--score--update--replan
loop. At information epoch \(\tau\), it archives

\[
p_\tau(\mathcal C)=p(\mathcal C\mid I_\tau)
\]

together with the exact information cutoff, evidence snapshot, model version,
parameters, random seed, and generated scenarios. New public evidence updates
the current calendar, while later realised outcomes are used to improve the
forecasting method itself.

Two updates are distinguished:

1. **Mission-specific evidence update:** a new date, assigned vehicle, payload
   arrival, added payload, cancellation, or launch occurrence changes the
   posterior for the affected mission immediately.
2. **Model calibration update:** archived forecasts are compared with realised
   outcomes so future occurrence, count, timing, site, orbit, and capacity
   distributions become better calibrated by provider, vehicle, site, orbit
   class, evidence class, and forecast lead time.

Every historical forecast must be retained. Without the forecast as it existed
at its information cutoff, final outcomes cannot reveal whether earlier
probabilities or intervals were calibrated. The archive records at least:

- Forecast identifier and accurate UTC `as_of` timestamp.
- Model and evidence-extractor versions.
- Source/evidence snapshot identifier.
- Scenario probabilities and random seed.
- Predicted event, count, timing, mark, and capacity distributions.
- Subsequent revisions and eventual realised outcome.
- Calibration scores and downstream SLA decisions.

Evaluation uses proper probabilistic scores rather than only point error:

- Brier score for occurrence or accessibility events.
- Log predictive density for counts and categorical marks.
- Continuous ranked probability score for launch time and capacity.
- Coverage of 50%, 80%, 90%, and 95% prediction intervals.
- Calibration by provider, vehicle, site, orbit class, evidence class, and lead
  time.
- Decision quality and regret for the resulting SLA commitments.

Immediate Bayesian conditioning may update an active forecast when evidence
arrives. Periodic recalibration compares archived forecasts with outcomes,
performs chronological backtests, and promotes a new model only when its
calibration and downstream decisions improve. A single unusual observation
must not silently replace the production model.

After every material forecast revision, the system regenerates launch
scenarios and reevaluates the accepted SLA portfolio. The product reports both
the reliability change and its cause, for example a primary-launch delay,
reduced accessible capacity, or loss of a compatible backup.

### 3.3 Simplified Bayesian launch and capacity predictor

The first product model uses only GCAT and periodic Launch Library 2 (LL2)
snapshots. GCAT supplies realised launches, payloads, vehicles, sites, and
orbits. Immutable LL2 snapshots record what was publicly announced at each
information cutoff. External market signals may be added only after they pass
chronological out-of-sample tests against these two baselines.

The forecast combines two processes:

1. **Announced missions:** for every mission visible at the information cutoff,
   estimate its probability of launch in each future time interval, delay
   beyond the horizon, or cancellation.
2. **Unannounced missions:** estimate provider-specific background launch
   counts so that a long-horizon forecast does not assume the current public
   manifest is complete.

For provider \(p\) and future interval \(t\), the initial count model is

\[
N_{p,t}\sim\operatorname{NegativeBinomial}(\lambda_{p,t},\phi),
\]

where the log intensity is a hierarchical Bayesian regression over provider,
season, recent cadence, announced missions, and any approved market signals.
The generator then samples coupled marks:

\[
p(s,v,o,m,c\mid p,t,I_\tau),
\]

where \(s\) is site, \(v\) vehicle, \(o\) broad orbit class, \(m\) manifested
demand, and \(c\) projected accessible capacity. Impossible provider--vehicle--
site combinations have zero probability. The first orbit classes are low-,
mid-, and polar/SSO LEO, MEO, GTO/GEO, and other.

Capacity is generated conditionally rather than independently:

\[
C_{\mathrm{residual}}=
\max\{0,C_{\mathrm{technical}}(v,o)-M_{\mathrm{manifested}}\},
\qquad
C_{\mathrm{accessible}}=A\,C_{\mathrm{residual}}.
\]

Vehicle guides provide a curated technical-capacity table. GCAT payload records
train the manifested-demand distribution. The accessibility factor \(A\) is
conditioned on mission/evidence class and is initially represented by explicit
conservative, central, and optimistic priors. Public data can validate
manifested demand and estimated physical residual, but usually cannot reveal
whether residual capacity was commercially bookable. Product output must
therefore call this quantity **projected accessible capacity**, show its
evidence class, and distinguish data from assumption.

The Bayesian posterior is updated as schedule revisions and realised launches
arrive. A separate residual-calibration layer corrects systematic errors in
occurrence probabilities, count distributions, timing quantiles, categorical
marks, and capacity intervals. Candidate market signals such as recent
satellite deployment counts, constellation activity, regulatory filings, or
provider backlog are admitted one at a time and retained only if they improve
chronological calibration and downstream SLA decisions relative to cadence and
announced-manifest baselines.

The first forecast output for every opportunity contains launch-time
distribution, provider, vehicle, site, orbit class, occurrence probability,
technical capacity, predicted manifested mass, projected accessible-capacity
distribution, evidence class, source snapshot, and model version.

### 3.4 SLA reliability, risk, and learning

Every prospective and accepted SLA receives both a reliability estimate and a
risk assessment. Reliability is

\[
R_i=P(\text{SLA }i\text{ is fulfilled}\mid I_\tau,\pi,\mathcal P),
\]

conditional on the current information, proposed policy \(\pi\), and accepted
portfolio \(\mathcal P\). Success is evaluated directly from the DSL outcome,
quantity, service-window, and completion requirements.

Risk combines failure probability with consequence:

\[
\operatorname{Risk}_i=
P(\text{failure}_i)E[L_i\mid\text{failure}_i],
\]

and reports expected loss, tail loss, and an explainable Low/Moderate/High/
Severe label. Loss may include penalties, lost revenue, emergency launch or
transfer cost, stranded inventory, business interruption, and harm to other
accepted SLAs. Scenario attribution identifies contributions from launch
occurrence, accessible capacity, orbit/transfer compatibility, inventory,
fleet availability, schedule, and shared-resource concentration. Marginal
probabilities must not be added as if independent.

The acceptance decision also reports incremental portfolio risk:

\[
\Delta\operatorname{Risk}_i=
\operatorname{Risk}(\mathcal P\cup\{i\})-
\operatorname{Risk}(\mathcal P).
\]

The risk layer improves through a forecast--decision--outcome loop. Each
decision archives the input snapshot, forecast/scenario version, proposed and
selected mitigation, predicted reliability and losses, risk drivers, quoted
terms, and portfolio state. Later observations record fulfilment, lateness,
failure cause, realised operational and contractual cost, mitigation use, and
effects on other SLAs. These data support recalibration of failure
probabilities, conditional loss distributions, driver attribution, and rating
thresholds.

Risk learning is separated into:

- **Physical/operational updating:** update launch, capacity, transfer,
  inventory, and service-duration models from their relevant observations.
- **Loss updating:** update consequence distributions from realised costs,
  claims, penalties, recovery actions, and business interruption.
- **Calibration:** compare predicted reliability and loss quantiles with
  outcomes by lead time, evidence class, service type, and risk tier.
- **Decision evaluation:** compare accepted mitigations and commitments with
  hindsight alternatives while avoiding training directly on the tool's own
  previous rating label.

Until sufficient SLA outcome and claims data exist, consequence values are
contractual or scenario assumptions. The interface must report an evidence-
confidence grade and uncertainty interval alongside the reliability and risk
label. A label is never shown without its probability, expected/tail loss,
main drivers, and mitigation sensitivity.

The learned predictive distribution is the primary forecast object. Calibrated
joint uncertainty sets over launch occurrence, time, orbit, and capacity are
derived from that distribution for robust planning and insurer-facing
resilience tiers.

## 4. Questions that must be resolved

### 4.1 Commercial questions

- What does an actual OOS SLA specify: outcome, service window, quantity, reliability, response time, remedy, or availability?
- Which failures trigger a contractual remedy, and which are force majeure?
- Is reliability priced as discrete service tiers or negotiated continuously?
- When can an accepted request still be moved between depots, vehicles, or launches?
- What reservation, cancellation, rebooking, and backup-launch terms are realistic?
- Does the provider control launch procurement, or is launch supplied by a partner/customer?

### 4.2 Operational questions

- Which OOS services share vehicles, tools, propellant, and spares?
- Can a service be split across visits or vehicles?
- When does a depot become operational?
- Which commodities are interchangeable and which are service-specific?
- What inspection, turnaround, maintenance, and contingency times apply?
- Which orbital transfers are feasible, and at what propellant/time cost?
- Which actions are reversible at each stage?

### 4.3 Uncertainty questions

- Which launch attributes are known, forecast, offered, contracted, and realised?
- How do launch time, insertion orbit, capacity, and price vary jointly?
- Are delays correlated by provider, vehicle family, pad, range, season, or shared manifest?
- How should unannounced future launch opportunities enter long-horizon scenarios?
- How is capacity availability represented when booking histories are unavailable?
- Which OOS uncertainties belong in the first model: service duration, consumption, vehicle failure, new SLA arrivals, or all of them later?

### 4.4 Decision-timing questions

- At what epochs is the problem solved?
- What information is available at each epoch?
- What is the booking/integration freeze point for a launch?
- What is the route/vehicle departure freeze point?
- Which recourse decisions may depend on which observations?
- How far should the commitment horizon extend?

## 5. Data requirements

Every field should be labelled as **observed**, **derived**, **predicted**, **contractual**, or **assumed**.

### 5.1 Launch-event backbone

Required fields:

- Stable launch identifier.
- Actual launch time and outcome.
- Provider, vehicle family and variant.
- Launch site and pad.
- Payloads and delivered mass.
- Insertion or destination orbit, with altitude, inclination, and plane where available.
- Vehicle reuse/recovery configuration where performance is affected.

Candidate sources:

- GCAT as the historical launch-event backbone.
- ESA DISCOS for cross-validation of launches, objects, mass, and orbit information.
- Provider user guides and NASA performance resources for vehicle capability.

### 5.2 Historical information sets

To backtest honestly, preserve what was known at each date:

- Observation timestamp.
- Advertised date/window and its precision.
- NET/TBD semantics.
- Advertised provider, vehicle, site, and orbit.
- Dated revisions, postponements, cancellations, and replacements.
- Source URL and retrieval timestamp.

Candidate sources:

- Launch Library 2 update histories and snapshots.
- Archived provider, spaceport, agency, and range announcements.
- A forward data-collection process that snapshots calendars regularly.

### 5.3 Capacity

Separate four quantities:

- Vehicle technical performance to a specified orbit.
- Payload mass actually carried.
- Capacity allocated to commercial rideshare or a logistics intermediary.
- Capacity offered to the OOS provider at a particular observation date.

Technical performance can be derived from vehicle and orbit records. Offered capacity requires provider/broker data or a stated market-demand model. Do not infer commercially available capacity by subtracting manifested payload mass from headline vehicle capacity.

If direct booking data remain unavailable, construct capacity scenarios using:

- Constellation deployment and replenishment demand.
- Historical operator-provider relationships.
- Satellite mass and orbit compatibility.
- Announced launch agreements and manifests.
- Booking deadlines, cancellations, and a provider acceptance rule.

Validate whether these features predict realised occupancy. Present the result as a demand-informed capacity model, not observed booking availability.

### 5.4 Cost

- Published dedicated-launch prices.
- Rideshare tariffs by mass and orbit.
- Integration, adapter, transfer-vehicle, and mission-management charges.
- Reservation, cancellation, and rebooking terms.
- Public contract awards, with included services recorded.
- Common currency and price-year conversion.

Store list prices, awarded contract values, and model assumptions separately.

### 5.5 OOS operations

- Candidate depot orbits and capacity.
- Depot dry mass and deployment requirements.
- Servicer dry mass, payload capacity, propulsion, and service capabilities.
- Commodity mass/volume, consumption, shelf life, and compatibility.
- Transfer time, delta-v, and propellant requirements.
- Service durations and turnaround times.
- Failure/maintenance assumptions.
- Initial and recurring costs.

Early values may come from literature and engineering case studies. Industry pilots should replace the most decision-sensitive assumptions.

### 5.6 SLA and market data

- Service request type and quantity.
- Client orbit and accessibility.
- Earliest/latest completion dates.
- Requested reliability.
- Revenue, lateness penalty, failure remedy, and cancellation terms.
- Customer priority and willingness to accept alternatives.
- Arrival process for future requests.

Because commercial SLA data will usually be confidential, begin with structured synthetic contracts and seek an anonymised industry case.

## 6. Literature map

### 6.1 OOS architecture and routing

**Role:** defines depot location, fleet sizing, client allocation, orbital accessibility, and servicing routes.

**Use in this work:** supplies the physical network and route-feasibility layer. The extension is gradual deployment, inventory, launch assignment, and explicit contractual commitments.

### 6.2 Space inventory and resupply

**Role:** models stock availability, replenishment, depot capacity, and service performance.

**Use in this work:** supplies inventory balance, resupply, and shortage logic. The extension is joint optimisation with deployment, launch capacity, and routing.

### 6.3 Dynamic space logistics networks

**Role:** represents infrastructure and commodity movement through time-expanded networks and captures deployment phases.

**Use in this work:** supplies the lifecycle representation linking Earth launch, orbital infrastructure, and operations.

### 6.4 Launch manifesting

Examples include Hwang and Ahn's freight-forwarder model.

**Role:** explains compatibility, carrier choice, multi-payload grouping, launch dates, and deployment sequences.

**Use in this work:** defines the market interface and feasible launch assignments. It is not the main product, and the first version need not reproduce a broker's internal manifest optimiser.

### 6.5 Revenue management and stochastic capacity allocation

Examples include Armar's cargo revenue-management work and the broader dynamic stochastic knapsack/revenue-management literature.

**Role:** explains request arrivals, protection levels, booking limits, cancellations, and the opportunity cost of scarce capacity.

**Use in this work:** supports a demand-informed model of external capacity consumption and, separately, the OOS provider's own SLA acceptance and pricing decisions.

### 6.6 Robust and adaptive robust optimisation

**Role:** protects decisions against sets of uncertain outcomes while allowing specified recourse after information is revealed.

**Use in this work:** protects accepted SLAs and frozen commitments across calibrated launch-calendar scenarios. Decision rules must respect information timing.

### 6.7 Data-driven uncertainty sets and conformal calibration

**Role:** converts historical forecast errors or generated scenarios into sets with testable out-of-sample coverage.

**Use in this work:** links a robustness setting to empirical launch-calendar coverage. If an uncertainty set has conditional coverage \(\alpha\), and an SLA is feasible throughout the set, \(\alpha\) provides a model-based lower bound on service reliability under the stated assumptions.

### 6.8 Stochastic and distributionally robust optimisation

**Role:** handles probability-weighted scenarios or uncertainty about their distribution.

**Use in this work:** provides important benchmarks. Experiments should determine whether adaptive robust protection creates better reliability-cost trade-offs than stochastic scheduling or deterministic buffers.

### 6.9 Rolling-horizon and model-predictive control

**Role:** repeatedly updates decisions as forecasts and system state change.

**Use in this work:** defines the operational workflow, state transition, commitment horizon, and freeze rules.

### 6.10 Service contracts and reliability engineering

**Role:** connects service outcomes, availability/reliability targets, penalties, and risk allocation.

**Use in this work:** makes the SLA representation commercially meaningful and prevents reliability from being treated solely as an optimisation parameter.

## 7. Proposed research sequence

### Study 1: Launch-calendar uncertainty

- Reconstruct historical information sets.
- Build the joint generative launch-access model over coupled time, site,
  vehicle, orbit, projected purchasable capacity, price, and booking deadline.
- Begin with realised-event and announced-calendar data, then add timestamped
  public releases and news as conditioning information.
- Compare point forecasts, factorised baselines, marked-event scenario
  generators, and calibrated uncertainty sets.
- Evaluate capacity forecasts separately by evidence class and avoid treating
  inferred residual capacity as an observed commercial offer.
- Evaluate chronological out-of-sample coverage and calibration.

### Study 2: OOS logistics under launch uncertainty

- Add depot/servicer deployment, inventory, resupply, and routing.
- Compare static, deterministic rolling-horizon, stochastic, and adaptive robust policies.
- Measure cost, SLA fulfilment, accepted demand, resource use, and recovery actions.

### Study 3: SLA acceptance and pricing

- Introduce prospective requests and accepted commitments.
- Compute incremental cost, capacity displacement, reliability, and risk premium.
- Test hard guarantees, priced violations, and tiered reliability products.

### Study 4: Industry case study

- Replace sensitive assumptions with an anonymised company case.
- Reproduce a historical or representative planning decision.
- Assess whether recommendations would have changed a booking, deployment, or customer-acceptance decision.

## 8. Validation design

### Forecast validation

- Use chronological train, calibration, and test periods.
- Prevent future announcements or final outcomes from leaking into earlier epochs.
- Evaluate calendar-level coverage, not only individual-field error.
- Check calibration by provider, orbit class, lead time, site, and season.
- Test rare disruptions and correlated delays separately.

### Optimisation validation

Compare against:

- Complete predeployment before service begins.
- Current-calendar deterministic planning.
- Fixed delay and capacity safety buffers.
- Stochastic programming using scenario probabilities.
- Adaptive robust scheduling.
- Perfect-information hindsight as an unattainable upper benchmark.

Report:

- SLA success and failure severity.
- Total and incremental lifecycle cost.
- Accepted SLA volume and revenue.
- Launch reservations, cancellations, and backups.
- Inventory shortages and stranded inventory.
- Fleet/depot utilisation.
- Regret relative to hindsight.
- Solve time and stability of recommendations.

## 9. Minimum viable product

### User workflow

1. Import current state and accepted SLAs.
2. Enter a prospective SLA.
3. Select or generate the current launch-access forecast.
4. Run acceptance and planning analysis.
5. Review a recommended commitment plan and alternatives.
6. Inspect why the request is risky and which constraint drives the result.
7. Export an auditable decision report.

### First outputs

- Accept/reject/defer recommendation.
- Feasible reliability tiers.
- Incremental cost and suggested minimum quote.
- Primary and backup launch reservations.
- Depot, vehicle, inventory, and route plan.
- Decisions to commit now versus keep provisional.
- Sensitivity to launch and capacity assumptions.
- Explanation of binding constraints and failure scenarios.

### Product requirement

Every recommendation must be reproducible from a versioned input snapshot, model version, solver configuration, and scenario/uncertainty-set version.

## 10. Recommended technology stack

### Research and modelling core

- **Python** for data preparation, forecasting, APIs, and integration.
- **Polars or pandas** for tabular processing; prefer Polars when event histories become large.
- **DuckDB and Parquet** for reproducible research snapshots and fast local analytics.
- **PostgreSQL + PostGIS** for the deployed system, including orbital/site metadata and temporal queries.
- **Pyomo** for the optimisation model if the team wants a mature Python-native algebraic modelling layer.
- **Gurobi** for production-scale mixed-integer optimisation when licensing permits; retain **HiGHS** for open development and continuous-integration tests.
- **scikit-learn, LightGBM/XGBoost, PyTorch, or probabilistic libraries** only as justified by forecasting experiments. Begin with interpretable baselines.
- **Astropy, poliastro, or validated internal orbital routines** for low/medium-fidelity compatibility calculations; isolate this behind a tested service so higher-fidelity tools can replace it.

Python is preferred over a split Python/Julia stack for the first product because it reduces integration and deployment complexity. JuMP remains attractive for solver research if computational evidence shows a material advantage.

### Service architecture

- **FastAPI** for typed internal and external APIs.
- Separate services/modules for data ingestion, scenario generation, optimisation, and reporting.
- **Pydantic** schemas for versioned inputs and outputs.
- **Celery/RQ or a managed job queue** for long-running optimisation jobs.
- **Redis** for job state and caching, if operational scale requires it.
- **Object storage** for immutable raw snapshots, scenario sets, and result bundles.

### User interface

- **React + TypeScript** for an industry-facing application.
- A lightweight **Streamlit** interface is acceptable for research interviews and the first demonstrator, but should not define the long-term architecture.
- Use maps/orbit visualisation sparingly; prioritise commitment timelines, scenario comparison, resource use, and decision explanations.

### Reproducibility and operations

- Git-based source control and tagged model releases.
- Containers for reproducible deployment.
- Data/model registry recording source, retrieval date, transformations, and licences.
- Automated schema, unit, integration, and small optimisation-regression tests.
- Structured logs, solver logs, run IDs, and audit trails.
- Role-based access, tenant isolation, encryption, and European deployment options for confidential commercial cases.

### Architecture principle

Keep the optimisation core independent of the interface and database. A solve request should be a versioned document, and a solve result should be another versioned document. This permits local, cloud, or customer-premises deployment without rewriting the mathematical model.

## 11. Industry discovery plan

### Interview groups

- OOS providers: SLA structure, mission commitments, deployment and recovery decisions.
- Orbital logistics providers: launch procurement, compatibility, capacity visibility, and rebooking.
- Satellite operators: service windows, willingness to pay, replacement alternatives, and reliability requirements.
- Launch aggregators: what an external customer can observe and commit to.
- Agencies/insurers/investors: evidence required to regard a service portfolio as credible.

### Questions to ask

- Describe the last decision where launch uncertainty changed a service or deployment plan.
- What was known, estimated, quoted, and contractually fixed at that moment?
- Which decision became irreversible first?
- What workaround was used, and what did it cost?
- Which output would have changed the decision?
- Which data could be shared in anonymised or aggregated form?
- What error would make the tool unacceptable?

### Desired outcome

Secure one design partner willing to provide:

- An anonymised planning case.
- Review of model assumptions and outputs.
- A short letter stating the decision problem is relevant.
- Ideally, a small historical dataset or synthetic case approved by the partner.

## 12. Productisation stages and gates

### Stage A: Research demonstrator

- One orbit regime, a small number of depots/vehicles, and simplified routes.
- Historical calendar reconstruction and a first joint generative
  launch-access model using transparent capacity assumptions and evidence
  classes.
- Public-information ingestion with retained timestamps, sources, and manual
  verification.
- Offline analyses and reproducible reports.

**Gate:** calibrated forecasts outperform simple baselines and change OOS decisions in meaningful test cases.

### Stage B: Design-partner pilot

- Company-specific state, contracts, and commitment rules.
- Scenario review and manual approval of recommendations.
- Secure deployment and audit logging.

**Gate:** planners judge the output credible and use it in at least one real planning exercise.

### Stage C: Operational decision support

- Automated data updates.
- Scheduled replanning and alerts.
- Integration with mission-planning and commercial systems.
- Human approval before bookings or customer commitments.

**Gate:** measured reduction in planning time, unnecessary reserves, or commitment risk.

## 13. Immediate next actions

1. Write a one-page definition of the OOS acceptance decision and identify the decision owner inside a company.
2. Create a five-screen mock-up before building the full optimiser.
3. Conduct five problem interviews: two OOS providers, two orbital-logistics/launch-integration providers, and one satellite operator or insurer.
4. Audit historical schedule-update coverage and destination-orbit completeness.
5. Define a transparent first capacity model with low, medium, and high market-pressure regimes.
6. Specify a small end-to-end case with one depot, two servicers, two commodities, and a portfolio of SLAs.
7. Implement deterministic rolling-horizon planning first, then add stochastic and robust variants as comparable layers.
8. Backtest the full decision pipeline rather than evaluating forecast accuracy alone.
9. Ask UAntwerp's Valorisation Office about IP, collaboration agreements, and a potential industry pilot before releasing substantial software publicly.
10. Approach Flanders Space and ESA BIC Belgium after the concept note and mock-up are ready.
