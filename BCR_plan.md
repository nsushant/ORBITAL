# BCR Model — Design Summary

## Formula

BCR = Benefits / Costs

## Benefits

Benefits = (n_serviced × V_sat) + max(0, 20000 − f3 × m_wet) × 3700

- n_serviced = 200 − f2 / 3  (from knee-point solution)
- V_sat = replacement value per serviced satellite (V1 or V2, see below)
- Leftover F9 mass = 20,000 kg − f3 × m_wet, sold at $3,700/kg market rate
- If fleet is too heavy to leave any spare capacity, rideshare revenue = 0

## Costs

Costs = servicer launch cost + servicer manufacturing cost + propellant cost

- Servicer launch:        f3 × m_wet × 3700         (full market rate, third-party operator)
- Servicer manufacturing: f3 × m_dry × c_mfg         (c_mfg is the free variable)
- Propellant:             m_prop_total × c_prop       (propellant-specific, from Table 2 Tirila 2023)

## Wet Mass

m_wet = m_dry × exp(5000 / (Isp × 9.80665))

- One tank (5000 m/s budget) per servicer launched
- Depot launched separately — its costs are excluded
- Wet mass is propellant-dependent (higher Isp → lighter at launch)
- m_dry fixed at 300 kg for the main analysis

## Satellite Replacement Values

Internal SpaceX launch rate assumed at 0.5 × 3700 = $1,850/kg

- V1: $770K manufacturing + 1850 × 260 kg = $1.251M
- V2: $1.77M manufacturing + 1850 × 800 kg = $3.25M

V1 vs V2 classification uses OBJECT_ID from the Celestrak catalog:
- V2 mini: launch year >= 2024, or (year == 2023 and launch number >= 29)
- All others: V1

## Knee-Point Selection

One solution per algorithm per trial is selected: the knee point of the 3D Pareto front.
Normalise f1, f2, f3 to [0,1] across the front, pick the point minimising Euclidean
distance to the ideal point (0, 0, 0).

## Main Output: Break-Even Manufacturing Cost

Setting BCR = 1 and solving for c_mfg:

c_mfg_max = (Benefits − servicer launch cost − propellant cost) / (f3 × m_dry)

- Positive c_mfg_max: mission can afford that $/kg of servicer hardware and still break even
- Viable range: [0, c_mfg_max]
- Plotted for each propellant, one result per algorithm knee-point

## Secondary Output: Dry Mass Sensitivity

Separate plot: c_mfg_max vs assumed dry mass, one curve per propellant.
Shows how affordable manufacturing cost shifts if the servicer is heavier or lighter.

## Falcon 9 Assumptions

- Launch cost: $74M
- Payload to LEO: 20,000 kg
- Market rideshare rate: $3,700/kg (= 74M / 20000)
- Internal SpaceX rate: $1,850/kg (= 0.5 × market rate)

## Propellant Costs and Isp (Tirila et al. 2023, Tables 2 and 5)

| Propellant | Isp (s) | Cost ($/kg) |
|------------|---------|-------------|
| Xe         | 2141    | 340         |
| Kr         | 2680    | 85          |
| I2         | 2178    | 32          |
| Bi         | 1697    | 8           |
| Zn         | 3034    | 3           |

## What Is NOT Included in Costs

- Depot launch or manufacturing cost
- Lost satellite penalty (unserviced sats reduce benefits but are not added to costs)
- Ground operations or mission control costs

## Pending Decision

Per-satellite assignment is not currently output by the Julia scheduler.
To compute exact BCR (summing actual V1/V2 values of serviced sats), the scheduler
needs to log which satellites are serviced per solution. This also opens the door to
making the scheduler value-aware (prioritising V2 over V1) and changing f2 from
unassigned time to unrecovered asset value. Decision deferred — to be discussed.
