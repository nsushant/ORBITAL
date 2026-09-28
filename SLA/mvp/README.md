# Local SLA portfolio demonstrator

Run from `C:\Users\snigudkar\ORBITAL`:

```powershell
python -m SLA.mvp.server
```

Or double-click `SLA\start_mvp.cmd`. Open `http://127.0.0.1:8765` while the launcher window remains open. The application has no web-framework dependency.

### Natural-language intake

When `OPENAI_API_KEY` is present, the chat uses the OpenAI Responses API with
strict structured output. The model extracts stated facts; deterministic code
validates them and constructs the planning object, and the planner computes
reliability. API responses are requested with `store: false`.

```powershell
$env:OPENAI_API_KEY = "your-project-api-key"
$env:OPENAI_SLA_MODEL = "gpt-5-mini"  # optional
python -m SLA.mvp.server
```

Without a key, the demo uses its offline guided parser.

The bundled example compares the accepted portfolio against the same portfolio plus a candidate SLA under three named uncertainty scenarios. The optimizer constructs a launch manifest from individual commodity and spacecraft payload candidates.

## Algorithms in the demonstrator

- An exact, dependency-free binary manifest MILP selects individual payloads, enforces each launch's mass capacity and respects item prerequisites. It is intended for MVP instances with at most 16 candidate items.
- A time-dependent transfer table is generated with the repository's Edelbaum thrust-coast-thrust model and J2 RAAN propagation for the relevant orbits, departure epochs and flight times.
- The ported MDLS runs inside every manifest/scenario evaluation and returns servicer routes, arrival times, transfer expenditure and unserved value.
- Commodity balances test whether routed services have the required inventory after scenario-specific launch success or failure.
- The selected manifest produces a deployment calendar, commodity resupply calendar, SLA reliability estimate, commercial value and next review time.

The current enumerated MILP is exact for the generated binary candidate set. A conventional MILP backend such as HiGHS or Gurobi is the scaling path for larger menus; it does not change the problem interface.

## Notebook demonstration

`../research/sla_portfolio_mvp_demo.ipynb` explains the example step by step, compares candidate manifests and plots sensitivity to launch-failure probability. Run the notebook from the ORBITAL repository root so `SLA` imports resolve normally.

## Current boundary

Explicit depot pickup visits inside MDLS routes, continuous payload sizing and launch-price curves remain later fidelity upgrades. The demonstrator currently treats each manifest candidate as an indivisible payload decision.
