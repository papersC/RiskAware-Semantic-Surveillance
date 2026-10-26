import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, json, time
from ras import *
out = {"gap": [], "scale": []}
for ncam in [8, 12, 16]:
    for seed in range(5):
        I = Instance(seed, n_cam=ncam, n_zone=ncam // 2, load=1.0)
        tr = I.scenarios(60, np.random.default_rng(1000 + seed))
        for lam in [0.0, 1.0]:
            t = time.time(); xg = ras_controller(I, tr, alpha=0.95, lam=lam); tg = time.time() - t
            x0 = ras_greedy(I, tr, lam=0.0, alpha=0.95); xo = ras_greedy(I, tr, lam=lam, alpha=0.95, init=x0) if lam > 0 else x0
            _, fg = solve_two_stage(I, tr, objective="mixed", lam=lam, alpha=0.95, fixed_x=xg, time_limit=120)
            _, fo = solve_two_stage(I, tr, objective="mixed", lam=lam, alpha=0.95, fixed_x=xo, time_limit=120)
            xm, fm = solve_two_stage(I, tr, objective="mixed", lam=lam, alpha=0.95, time_limit=300, mip_gap=1e-4)
            out["gap"].append(dict(ncam=ncam, seed=seed, lam=lam, obj_g=fg["obj"], obj_static=fo["obj"], obj_m=fm["obj"], t_g=tg, t_m=fm["time"], status=int(fm["status"])))
            print(ncam, seed, lam, round(fg["obj"] - fm["obj"], 4), round(fo["obj"] - fm["obj"], 4), round(tg, 2), round(fm["time"], 1), fm["status"], flush=True)
for ncam in [12, 24, 48, 72, 96]:
    for seed in range(3):
        I = Instance(seed, n_cam=ncam, n_zone=ncam // 2, load=1.0)
        tr = I.scenarios(300, np.random.default_rng(1000 + seed))
        t = time.time(); xg = ras_controller(I, tr, alpha=0.95, lam=1.0); tg = time.time() - t
        t = time.time(); _ = recourse(I, xg, tr[0]); tr_t = time.time() - t
        out["scale"].append(dict(ncam=ncam, seed=seed, t_g=tg, t_rec=tr_t, nd=I.nd))
        print("scale", ncam, seed, round(tg, 2), round(tr_t, 4), flush=True)
json.dump(out, open("v2_scale.json", "w"))
