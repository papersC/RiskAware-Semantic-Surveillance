# RiskAware-Semantic-Surveillance (RAS-6G)

Simulator, experiments and LaTeX sources for:

**Risk-Aware Semantic Utility Optimization for Resilient Multi-Camera Surveillance in 6G Edge Networks**  
(Nedal Ababneh, Khaled Almi'ani, Ammar Kamal Abasi) — IEEE TNSM submission.

## Reproduce experiments

```bash
cd code
pip install -r requirements.txt
python3 calibrate_kappa.py
for k in main abl pareto rho shift ssize load sev large; do python3 run_all.py $k; done
python3 run_scale2.py
for k in matched compliance measured approx; do python3 run_extra.py $k; done
python3 make_figures.py
```

JSON results live in `data/`. Full suite takes on the order of a few hours on two cores.

## Manuscript

From the repository root:

```bash
latexmk -pdf main.tex
```

## Notes

Synthetic flow-level results. Literature-inspired profile numbers are documented in the manuscript and `code/run_extra.py`.
