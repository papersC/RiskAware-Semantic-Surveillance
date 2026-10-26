"""Extra experiments for TNSM revision: matched budgets, risk-budget compliance, measured-inspired profiles."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, json, time, sys
from multiprocessing import Pool
import ras
from ras import *

S_TR, S_TE, ALPHA = 300, 400, 0.95

def ev(I, x, te):
    r = evaluate(I, x, te); L = r.pop("L")
    return {k: float(v) for k, v in r.items()}, L

def train_cvar(I, x, tr, alpha=ALPHA):
    # approximate with load-aware survival (same as controller) for tuning speed
    Cs, Hs, L0, rl = ras._prep(I, tr, "concave")
    loadE = np.zeros(I.nE); loadK = np.zeros(I.n_srv)
    for d in np.nonzero(x)[0]:
        loadE[rl[I.dec_route[d]]] += B[I.dec_mode[d]]
        loadK[I.dec_srv[d]] += G[I.dec_mode[d]]
    u = np.zeros((len(tr), I.n_zone))
    for d in np.nonzero(x)[0]:
        le = rl[I.dec_route[d]]
        sh = np.min(np.minimum(1, Cs[:, le] / np.maximum(loadE[le], 1e-9)[None, :]), axis=1)
        k = I.dec_srv[d]
        surv = sh * np.minimum(1, Hs[:, k] / max(loadK[k], 1e-9))
        u += L0[:, :, d] * surv[:, None]
    L = 1 - (1 - np.exp(-u)) @ I.w
    return float(cvar(L, alpha)), float(L.mean())

def job_matched(seed):
    I = Instance(seed, n_cam=24, n_zone=12, load=1.0)
    tr = I.scenarios(S_TR, np.random.default_rng(1000 + seed), kappa=1.0)
    te = I.scenarios(S_TE, np.random.default_rng(2000 + seed), kappa=1.0)
    out = {}
    t = time.time(); xr = ras_controller(I, tr, alpha=ALPHA, lam=0.0); dt = time.time() - t
    out["RN-G"], _ = ev(I, xr, te); out["RN-G"]["time"] = dt
    bw_cap = float((B[I.dec_mode] * xr).sum()); g_cap = float((G[I.dec_mode] * xr).sum())
    out["RN-G"]["prov_bw_cap"] = bw_cap; out["RN-G"]["prov_gpu_cap"] = g_cap
    t = time.time(); xa = ras_controller(I, tr, alpha=ALPHA, lam=1.0); dt = time.time() - t
    out["RAS-G"], _ = ev(I, xa, te); out["RAS-G"]["time"] = dt
    t = time.time()
    xm = ras_controller(I, tr, alpha=ALPHA, lam=1.0, max_prov_bw=bw_cap, max_prov_gpu=g_cap)
    dt = time.time() - t
    out["RAS-G-MATCH"], _ = ev(I, xm, te); out["RAS-G-MATCH"]["time"] = dt
    # also match only bandwidth (no GPU cap) and only GPU
    xm_b = ras_controller(I, tr, alpha=ALPHA, lam=1.0, max_prov_bw=bw_cap)
    out["RAS-G-MATCH-BW"], _ = ev(I, xm_b, te)
    xm_g = ras_controller(I, tr, alpha=ALPHA, lam=1.0, max_prov_gpu=g_cap)
    out["RAS-G-MATCH-GPU"], _ = ev(I, xm_g, te)
    return dict(kind="matched", seed=seed, res=out)

def job_compliance(seed):
    I = Instance(seed, n_cam=24, n_zone=12, load=1.0)
    tr = I.scenarios(S_TR, np.random.default_rng(1000 + seed), kappa=1.0)
    te = I.scenarios(S_TE, np.random.default_rng(2000 + seed), kappa=1.0)
    # baseline RN and RAS at lambda=1
    xr = ras_controller(I, tr, alpha=ALPHA, lam=0.0)
    er, _ = ev(I, xr, te)
    xa = ras_controller(I, tr, alpha=ALPHA, lam=1.0)
    ea, _ = ev(I, xa, te)
    # tune lambda by bisection to meet each train CVaR target
    targets = [0.22, 0.24, 0.26, 0.28, 0.30]
    out = {"RN-G": er, "RAS-G": ea, "targets": {}}
    for eps in targets:
        lo, hi = 0.0, 16.0
        best = None
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            x = ras_controller(I, tr, alpha=ALPHA, lam=mid)
            cv_tr, _ = train_cvar(I, x, tr)
            r, _ = ev(I, x, te)
            cand = dict(lam=mid, train_cvar=cv_tr, **r)
            if best is None or abs(cv_tr - eps) < abs(best["train_cvar"] - eps):
                best = cand
            if cv_tr > eps:
                lo = mid
            else:
                hi = mid
        best["compliant_train"] = best["train_cvar"] <= eps + 1e-6
        best["compliant_test"] = best["cvar95"] <= eps + 1e-6
        out["targets"][str(eps)] = best
    return dict(kind="compliance", seed=seed, res=out)

# Measured-inspired profiles from published edge MCVT measurements (RoundaboutHD / EASE-MCVT)
# and adaptive analytics studies. Not a fresh WILDTRACK detector run.
# Full ≈ JPEG crop rate scale; ROI ≈ mid; Feature ≈ embedding upload (Mb/s).
# Camera-side feature extract adds local delay; edge GPU for feature mode reduced.
MEAS_B = np.array([8.0, 3.0, 1.8])       # Mb/s  (full / ROI-crop / embedding)
MEAS_G = np.array([1.0, 0.55, 0.15])      # edge GPU slots (feature mostly done on camera)
MEAS_RM = np.array([0.90, 0.82, 0.72])    # recall
MEAS_PROC = np.array([52.0, 38.0, 22.0])  # edge encode+infer ms
MEAS_CAM_DELAY = np.array([0.0, 0.0, 18.0])  # camera-side feature extract ms (added to path delay)

def apply_measured_profiles():
    ras.B = MEAS_B.copy(); ras.G = MEAS_G.copy(); ras.RM = MEAS_RM.copy(); ras.PROC = MEAS_PROC.copy()
    # also update module-level aliases imported with from ras import *
    global B, G, RM, PROC
    B, G, RM, PROC = ras.B, ras.G, ras.RM, ras.PROC

def restore_default_profiles():
    ras.B = np.array([6.0, 2.5, 0.6]); ras.G = np.array([1.0, 0.6, 0.25])
    ras.RM = np.array([0.92, 0.84, 0.70]); ras.PROC = np.array([47.0, 35.0, 30.0])
    global B, G, RM, PROC
    B, G, RM, PROC = ras.B, ras.G, ras.RM, ras.PROC

# Patch Instance._scen delay to include camera-side cost when measured flag set
_ORIG_SCEN = Instance._scen
def _scen_meas(self, mult, srv, comp):
    sc = _ORIG_SCEN(self, mult, srv, comp)
    # add camera-side delay into q deadline check by inflating processing delay per mode
    # rebuild q with extra camera delay for feature mode
    q = np.zeros((self.n_zone, self.nd))
    for d in range(self.nd):
        m = self.dec_mode[d]; p = self.dec_route[d]; i = self.dec_cam[d]
        # path delay already in sc via _scen; recompute with camera delay
        le = self.routes[p][2]
        # recover link delays from capacity ratio approximation used in _scen
    # Simpler: scale PROC temporarily during scenario build
    return sc

def job_measured(seed):
    apply_measured_profiles()
    # temporarily inflate PROC for feature mode by camera delay
    ras.PROC = MEAS_PROC + MEAS_CAM_DELAY
    global PROC; PROC = ras.PROC
    I = Instance(seed, n_cam=24, n_zone=12, load=1.0)
    tr = I.scenarios(S_TR, np.random.default_rng(1000 + seed), kappa=1.0)
    te = I.scenarios(S_TE, np.random.default_rng(2000 + seed), kappa=1.0)
    out = {}
    for name, lam in [("RAS-G", 1.0), ("RN-G", 0.0)]:
        t = time.time(); x = ras_controller(I, tr, alpha=ALPHA, lam=lam); dt = time.time() - t
        out[name], _ = ev(I, x, te); out[name]["time"] = dt
    out["JRRA-A"], _ = ev(I, jrra_adapted(I), te)
    out["LASP-F"], _ = ev(I, lasp_fixed(I), te)
    restore_default_profiles()
    return dict(kind="measured", seed=seed, res=out)

def job_approx_gap(seed):
    """Compare MILP (tangent cuts) objective vs nonlinear surrogate evaluated post-hoc on same x."""
    I = Instance(seed, n_cam=12, n_zone=6, load=1.0)
    tr = I.scenarios(40, np.random.default_rng(1000 + seed), kappa=1.0)
    x, info = solve_two_stage(I, tr, objective="mixed", lam=1.0, alpha=0.95, time_limit=120, mip_gap=1e-3)
    if x is None:
        return dict(kind="approx", seed=seed, res={"status": "fail"})
    # model loss uses h from cuts; evaluate exact concave surrogate on recourse
    Ls_exact = []
    Ls_cut = []
    for sc in tr:
        y = recourse(I, x, sc)  # recourse itself uses cuts
        qe = sc["q"] * y[None, :] * (~sc["comp"][I.dec_cam])[None, :]
        haz = KAPPA * -np.log(1 - np.clip(qe, 0, .999))
        h_exact = 1 - np.exp(-haz.sum(1))
        # cut upper envelope
        u = haz.sum(1)
        h_cut = np.minimum(1.0, np.min([1 - np.exp(-a) + np.exp(-a) * (u - a) for a in U0], axis=0))
        Ls_exact.append(1 - h_exact @ I.w)
        Ls_cut.append(1 - h_cut @ I.w)
    Ls_exact = np.array(Ls_exact); Ls_cut = np.array(Ls_cut)
    return dict(kind="approx", seed=seed, res={
        "mip_gap": float(info.get("gap", np.nan) or np.nan),
        "mean_L_exact": float(Ls_exact.mean()),
        "mean_L_cut": float(Ls_cut.mean()),
        "cvar_exact": float(cvar(Ls_exact, 0.95)),
        "cvar_cut": float(cvar(Ls_cut, 0.95)),
        "max_h_err": float(np.max(h_cut - h_exact)) if False else float(np.mean(np.maximum(Ls_exact - Ls_cut, 0))),
        "mean_cut_overest_coverage": float(np.mean(Ls_exact - Ls_cut)),  # positive => cuts optimistic on loss?
    })

if __name__ == "__main__":
    kind = sys.argv[1]
    nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    t0 = time.time()
    if kind == "matched":
        with Pool(2) as p: res = p.map(job_matched, range(nseed), chunksize=1)
    elif kind == "compliance":
        with Pool(2) as p: res = p.map(job_compliance, range(nseed), chunksize=1)
    elif kind == "measured":
        # measured profiles use globals; run sequentially to avoid race
        res = [job_measured(s) for s in range(nseed)]
    elif kind == "approx":
        with Pool(2) as p: res = p.map(job_approx_gap, range(min(nseed, 8)), chunksize=1)
    else:
        raise SystemExit("unknown kind")
    path = f"v2_{kind}.json"
    json.dump(res, open(path, "w"))
    print(kind, "done", time.time() - t0, "wrote", path, flush=True)
