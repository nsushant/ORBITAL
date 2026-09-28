# Post-MVP deployment and data infrastructure

This document deliberately defers work that is not required to demonstrate the
Bayesian launch-capacity learning method in a research notebook.

## Deferred ingestion infrastructure

- Scheduled LL2 and GCAT refreshes with failure alerts and replay.
- Continuous document and news discovery.
- An LLM extraction service that converts dated releases, filings, manifests,
  launch-provider pages, and news into cited structured evidence.
- Human review queues for ambiguous entities and low-confidence claims.
- A maintained LL2-to-GCAT mission crosswalk.
- Source licensing, access-control, and retention enforcement.

## Deferred production data platform

- Storage monitoring and automated enforcement of the 10 GB local limit.
- Cloud object storage for immutable raw evidence.
- Managed PostgreSQL, backups, schema evolution, and disaster recovery.
- Job orchestration, observability, secrets management, and audit logging.
- Dataset and model version registries.

## Deferred predictive components

- Empirically fitted booking curves from real, time-stamped booking or manifest
  histories.
- Certified vehicle capacity surfaces by orbit, inclination, recovery mode,
  rideshare hardware, volume, and integration constraints.
- Delay, cancellation, provider reliability, and destination-orbit models.
- Cue-detection and false-positive calibration for public evidence.
- Joint dependence across launches, providers, customers, and constellations.
- Insurance loss-severity and portfolio-accumulation models.

## Deferred product capabilities

- SLA ingestion and compatibility screening.
- Risk pricing, explanation, and underwriter overrides.
- Scenario APIs, dashboards, user accounts, and role-based access.
- Continuous calibration reports and model-drift monitoring.
- Commercial data agreements with launch providers, aggregators, insurers, and
  constellation operators.

## Re-entry criterion

Return to this work after the notebook demonstrates that posterior updates and
uncertainty sets improve chronological predictive coverage, or when a partner
can provide real booking/manifest revision data. Until then, use the fixed GCAT
snapshot and clearly labelled synthetic booking histories.

## Candidate: local lightweight semantic evidence layer

This is the preferred candidate for the research demonstrator when document
language must be converted into weighted Bayesian evidence without Hugging Face,
a hosted model API, or a model hub in the runtime toolchain.

### Responsibilities

1. Extract candidate sentences concerning manifests, payload counts, booked or
   available capacity, launch dates, destination orbits, delays, cancellations,
   providers, customers, and contracts.
2. Parse quantities and their mathematical relation deterministically:
   `exact`, `lower_bound`, `upper_bound`, `interval`, or `approximate`.
3. Classify epistemic status rather than sentiment. Candidate labels are
   `realized`, `confirmed`, `contracted`, `planned`, `targeted`, `possible`,
   `speculative`, and `cancelled`.
4. Extract source role and claim scope, including whether a statement describes
   the whole launch, one integrator, one customer, or one payload group.
5. Group repeated reporting of the same underlying announcement so copied news
   does not create multiple independent Bayesian updates.
6. Emit cited, timestamped structured evidence for the observation model.

### Candidate implementation

- Python regular expressions and explicit parsing rules for dates, quantities,
  units, bounds, and common modal phrases.
- A local scikit-learn TF-IDF plus logistic-regression classifier for epistemic
  status. Word and character n-grams provide a small, fast, inspectable model.
- TF-IDF cosine similarity for local duplicate-claim grouping.
- Platt scaling or isotonic calibration after enough labelled examples exist.
- JSONL and PostgreSQL records for source text, extracted claims, corrections,
  classifier version, and provenance.
- `joblib` for storing the fitted classifier locally.

No network connection or external model service is required during inference.
A small manually labelled corpus can initialize the classifier, and corrected
production examples can be used for later retraining.

### Separation between semantics and statistical weight

The classifier identifies what the language means. It does not directly decide
how much the Bayesian model should trust the claim. A transparent rule maps
semantic attributes to an initial evidence weight:

```text
weight = source_weight × commitment_weight × scope_weight × specificity_weight
```

The evidence record must retain both the relation and the weight. For example,
"Exolaunch will deploy 49 satellites" is a strong lower bound on Exolaunch's
portion of the manifest; it is not a weak equality claim about the whole launch.
Initial weights are expert assumptions and must be tested through sensitivity
analysis. Later versions can calibrate them using historical claims and realized
manifests.

### Minimal dependencies

- Python standard library
- scikit-learn
- joblib
- optionally BeautifulSoup for local HTML-to-text conversion

### Optional fallback

If rules and the classifier cannot resolve complex passages or cross-sentence
references, an optional local model served through `llama.cpp` and a fixed GGUF
file can emit the same evidence schema. This fallback is outside the initial
research demonstrator and does not change the downstream Bayesian interface.

### Evaluation

Evaluate extraction separately from launch-capacity prediction:

- quantity and relation accuracy;
- epistemic-class precision, recall, and calibration;
- scope accuracy;
- duplicate-grouping accuracy;
- downstream sensitivity of posterior capacity and SLA risk to evidence weights.
