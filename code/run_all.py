import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, json, time, sys
from multiprocessing import Pool
import ras
from ras import *
S_TR, S_TE, ALPHA = 300, 400, 0.95

def make(I, tr, nom):
    shortest = np.array([min(I.cam_routes[i], key=lambda p: len(I.routes[p][2])) for i in range(I.n_cam)])
    def oldg():
        x0 = ras_greedy(I, tr, lam=0.0, alpha=ALPHA); return ras_greedy(I, tr, lam=1.0, alpha=ALPHA, init=x0)
    def gain_only():
        x0, _ = ras_greedy2(I, tr, alpha=ALPHA, lam=0.0, rule="gain"); return ras_greedy2(I, tr, alpha=ALPHA, lam=1.0, rule="gain", init=x0)[0]
    return {
        "RAS-G": lambda: ras_controller(I, tr, alpha=ALPHA, lam=1.0),
        "RN-G": lambda: ras_controller(I, tr, alpha=ALPHA, lam=0.0),
        "NOM-G": lambda: ras_controller(I, nom, alpha=ALPHA, lam=0.0),
        "ADD-G": lambda: ras_controller(I, tr, alpha=ALPHA, lam=1.0, surrogate="add"),
        "JRRA-A": lambda: jrra_adapted(I),
        "LASP-F": lambda: lasp_fixed(I),
        "NOROUTE": lambda: ras_controller(I, tr, alpha=ALPHA, lam=1.0, mask=np.isin(I.dec_route, shortest)),
        "FULLONLY": lambda: ras_controller(I, tr, alpha=ALPHA, lam=1.0, mask=I.dec_mode == 0),
        "STATIC": oldg,
        "GAINONLY": gain_only,
    }

def ev(I, x, te):
    r = evaluate(I, x, te); L = r.pop("L"); return {k: float(v) for k, v in r.items()}, L

def job(args):
    kind, seed, val = args
    load, kd, ncam, S = 1.0, 1.0, 24, S_TR
    if kind == "load": load = val
    if kind == "sev": kd = val
    if kind == "large": ncam = 48
    I = Instance(seed, n_cam=ncam, n_zone=ncam // 2, load=load)
    rng_tr = np.random.default_rng(1000 + seed)
    tr = I.scenarios(S, rng_tr, kappa=kd)
    te = I.scenarios(S_TE, np.random.default_rng(2000 + seed), kappa=kd)
    nom = I.scenarios(1, np.random.default_rng(1), nominal=True)
    M = make(I, tr, nom); out = {}; tails = {}
    names = ["RAS-G", "RN-G", "NOM-G", "ADD-G", "JRRA-A", "LASP-F"]
    if kind == "abl": names = ["RAS-G", "RN-G", "ADD-G", "NOROUTE", "FULLONLY", "NOM-G", "STATIC", "GAINONLY"]
    if kind == "pareto":
        names = []
        for lam in [0, 0.25, 0.5, 1, 2, 4, 8]:
            t = time.time(); x = ras_controller(I, tr, alpha=ALPHA, lam=lam); dt = time.time() - t
            r, _ = ev(I, x, te); r["time"] = dt; out[f"lam{lam}"] = r
    if kind == "ssize":
        names = []
        for Sn in [25, 50, 100, 200, 300, 500]:
            trn = I.scenarios(Sn, np.random.default_rng(5000 + seed), kappa=kd)
            t = time.time(); x = ras_controller(I, trn, alpha=ALPHA, lam=1.0); dt = time.time() - t
            r, _ = ev(I, x, te); r["time"] = dt; out[f"S{Sn}"] = r
    if kind == "shift":
        names = []
        xs = {"RAS-G": M["RAS-G"](), "RN-G": M["RN-G"](), "JRRA-A": M["JRRA-A"](), "LASP-F": M["LASP-F"]()}
        for kt in [1.0, 1.5, 2.0, 2.5]:
            tek = I.scenarios(S_TE, np.random.default_rng(3000 + seed), kappa=kt)
            for n, x in xs.items():
                out[f"{n}@{kt}"], _ = ev(I, x, tek)
            trk = I.scenarios(S, np.random.default_rng(4000 + seed), kappa=kt)
            out[f"RAS-G(matched)@{kt}"], _ = ev(I, ras_controller(I, trk, alpha=ALPHA, lam=1.0), tek)
    if kind == "rho":
        names = []
        xs = {"RAS-G": M["RAS-G"](), "RN-G": M["RN-G"](), "ADD-G": M["ADD-G"](), "JRRA-A": M["JRRA-A"](), "LASP-F": M["LASP-F"]()}
        for rho in [0.0, 0.2, 0.4, 0.6, 0.8]:
            ras.RHO = rho
            for n, x in xs.items(): out[f"{n}@{rho}"], _ = ev(I, x, te)
        ras.RHO = 0.4
    xs_main = {}
    for n in names:
        t = time.time(); x = M[n](); dt = time.time() - t
        xs_main[n] = x
        out[n], L = ev(I, x, te); out[n]["time"] = dt
        if kind in ("main", "large"): tails[n] = [float(v) for v in L]
    if kind == "main":
        ys = oracle(I, te[:100])
        r = evaluate(I, None, te[:100], ys=ys); L = r.pop("L")
        out["ORACLE@100"] = {k: float(v) for k, v in r.items()}; tails["ORACLE@100"] = [float(v) for v in L]
        for n in ["RAS-G", "RN-G", "JRRA-A"]:
            r, L = ev(I, xs_main[n], te[:100]); out[n + "@100"] = r
    return dict(kind=kind, seed=seed, val=val, res=out, tails=tails)

if __name__ == "__main__":
    kind = sys.argv[1]
    nseed = 15 if kind not in ("large",) else 8
    vals = {"load": [1.0, 2.0, 3.0, 4.0, 5.0], "sev": [0.5, 1.0, 1.5, 2.0, 2.5]}.get(kind, [1.0])
    jobs = [(kind, s, v) for v in vals for s in range(nseed)]
    t = time.time()
    with Pool(2) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"v2_{kind}.json", "w"))
    print(kind, "done", time.time() - t, flush=True)
