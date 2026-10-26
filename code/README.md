# RAS-6G simulator and experiment scripts

This folder contains the code that produced every number, table and figure in the manuscript
"Risk-Aware Semantic Utility Optimization for Resilient Multi-Camera Surveillance in 6G Edge Networks".

## Files

| File | Purpose |
|------|---------|
| `ras.py` | Instance and disruption-scenario generator, two-stage MILP (HiGHS via SciPy), frame-shedding recourse LP, Gaussian-copula detection model, RAS-G controller (`ras_controller`, `ras_greedy2`), static-survival variant (`ras_greedy`), baselines JRRA-A (`jrra_adapted`) and LASP-F (`lasp_fixed`), clairvoyant oracle |
| `run_all.py` | All experiments of Section VI. Usage: `python3 run_all.py <kind>` with kind in `main abl pareto rho shift ssize load sev large`. Writes `v2_<kind>.json` (copy to `../data/`) |
| `run_extra.py` | Matched-budget, risk-budget compliance, literature-inspired profiles, cut-vs-nonlinear gap (`matched`, `compliance`, `measured`, `approx`) |
| `run_scale2.py` | Optimality gap against the exact MILP (8, 12, 16 cameras) and runtime scaling (12 to 96 cameras). Writes `v2_scale.json` |
| `calibrate_kappa.py` | Calibration of the overlap factor kappa = 0.87, the surrogate RMSE values and the tangent-cut error bound (Section III, Proposition 2) |
| `make_figures.py` | Regenerates the pgfplots figure files in `../figures/`, the tables in `../tables/` and `../data/stats.json` (means, confidence intervals, Wilcoxon and Holm-adjusted tests) from the JSON results in `../data/` |
| `requirements.txt` | Python dependencies |

## Reproduction

```
pip install -r requirements.txt
python3 calibrate_kappa.py
for k in main abl pareto rho shift ssize load sev large; do python3 run_all.py $k; done
python3 run_scale2.py
for k in matched compliance measured approx; do python3 run_extra.py $k; done
python3 make_figures.py
```

All random streams are seeded (topology seed s, training scenarios 1000+s, test scenarios 2000+s),
and BLAS is forced to a single thread at import time. Under these settings, repeated runs reproduce the reported numbers exactly on the same platform.
The full set of experiments takes about three hours on a two-core machine.

## Model summary

* 24 cameras, 12 zones, 6 access points, 3 edge servers by default (Table II of the paper).
* Modes: full (6 Mb/s, recall 0.92), ROI (2.5 Mb/s, 0.84), feature (0.6 Mb/s, 0.70). These are synthetic profiles.
* Disruptions: log-normal shadowing, regional access-point outage, mesh-link failure, server overload and feed suppression.
* Evaluation: event detection per zone with a Gaussian copula (correlation 0.4), CVaR at 95%.

The simulator is flow-level. It does not model packets, transport protocols or radio scheduling.
