"""Calibration of the overlap factor kappa of the concave coverage surrogate (Section III).

Draws 4000 random view sets (1 to 4 views, per-view detection probability U(0.3, 0.95)),
computes the Gaussian-copula detection probability (correlation 0.4) and fits kappa in
P(detect) ~ 1 - exp(-kappa * sum(-ln(1 - q))). Also reports the RMSE of the independence
model and of the capped additive surrogate, and the maximum error of the ten tangent cuts.
"""
import numpy as np
from ras import detect_prob, U0

rng = np.random.default_rng(0)
data = []
for _ in range(4000):
    n = rng.integers(1, 5); q = rng.uniform(0.3, 0.95, n); data.append((q, detect_prob(q)))
def rmse(f): return float(np.sqrt(np.mean([(f(q) - p) ** 2 for q, p in data])))
grid = np.round(np.arange(0.70, 1.0001, 0.01), 2)
errs = {k: rmse(lambda q, k=k: 1 - np.exp(-k * (-np.log(1 - q)).sum())) for k in grid}
kbest = min(errs, key=errs.get)
print(f"best kappa = {kbest:.2f}, RMSE = {errs[kbest]:.4f}")
print(f"independence RMSE = {rmse(lambda q: 1 - np.prod(1 - q)):.4f}")
print(f"capped additive RMSE = {rmse(lambda q: min(1.0, q.sum())):.4f}")
u = np.linspace(0, 20, 200001); f = 1 - np.exp(-u)
cuts = np.minimum(np.min([1 - np.exp(-a) + np.exp(-a) * (u - a) for a in U0], axis=0), 1)
print(f"max tangent-cut error = {(cuts - f).max():.4f} at u = {u[(cuts - f).argmax()]:.2f}")
