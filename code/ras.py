"""RAS-6G simulator: risk-aware semantic utility orchestration for multi-camera surveillance.

Flow-level, scenario-based evaluation. All parameters are synthetic and documented in Table II of the paper.
Key entry points: Instance, Instance.scenarios, solve_two_stage (exact MILP), ras_controller (RAS-G),
ras_greedy (static-survival variant used in the ablation), jrra_adapted, lasp_fixed, recourse, evaluate, oracle.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")        # single-threaded BLAS for bitwise reproducibility
import numpy as np, itertools, time
import networkx as nx
from scipy.optimize import milp, linprog, LinearConstraint, Bounds
from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import norm

MODES = ["F", "R", "X"]                    # full frame, ROI, features
B = np.array([6.0, 2.5, 0.6])              # Mb/s per mode
G = np.array([1.0, 0.6, 0.25])             # GPU slot units per mode
RM = np.array([0.92, 0.84, 0.70])          # base detection recall per mode
PROC = np.array([47.0, 35.0, 30.0])        # encode + inference delay (ms) at nominal load
D_CRIT, D_NORM = 80.0, 150.0
RHO = 0.4                                  # inter-view detection correlation (Gaussian copula)
KAPPA = 0.87                               # correlation-calibration factor of the concave surrogate
U0 = np.array([0, .1, .25, .5, .75, 1, 1.5, 2, 3, 4.5])
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(24); GH_W = GH_W / GH_W.sum()


class Instance:
    def __init__(self, seed, n_cam=24, n_zone=12, load=1.0, n_srv=None, P=3, gpu_load=0.75):
        rng = np.random.default_rng(seed)
        self.rng = rng; self.n_cam = n_cam; self.n_zone = n_zone
        zc = rng.uniform(10, 90, (n_zone, 2))
        # cameras: anchored near zones so most zones are seen by 2-3 cameras
        anchor = np.concatenate([np.arange(n_zone), rng.integers(0, n_zone, max(0, n_cam - n_zone))])[:n_cam]
        cc = zc[anchor] + rng.normal(0, 9, (n_cam, 2))
        R = 30.0
        d = np.linalg.norm(cc[:, None, :] - zc[None, :, :], axis=2)
        V = np.where(d < R, 0.5 + 0.5 * (1 - d / R), 0.0)
        V *= (rng.uniform(size=V.shape) < 0.8) | (d < 12)       # occlusion: some in-range views unusable
        V[np.arange(n_cam), anchor] = np.maximum(V[np.arange(n_cam), anchor], 0.75)
        self.V = V
        n_crit = max(1, int(round(0.25 * n_zone)))
        crit = np.zeros(n_zone, bool); crit[rng.choice(n_zone, n_crit, replace=False)] = True
        w = np.where(crit, 4.0, 1.0); self.w = w / w.sum(); self.crit = crit
        self.D = np.where(crit, D_CRIT, D_NORM)
        # network: access nodes on a jittered grid, k-nearest mesh, 3 gateways to edge servers
        n_acc = max(4, int(np.ceil(n_cam / 4)))
        side = int(np.ceil(np.sqrt(n_acc)))
        acc = np.array([[((i % side) + .5) * 100 / side, ((i // side) + .5) * 100 / side] for i in range(n_acc)])
        acc += rng.normal(0, 4, acc.shape)
        self.n_acc = n_acc
        n_srv = n_srv or max(2, int(round(n_cam / 8)))
        self.n_srv = n_srv
        Gr = nx.Graph()
        da = np.linalg.norm(acc[:, None] - acc[None], axis=2)
        for a in range(n_acc):
            for b in np.argsort(da[a])[1:4]:
                Gr.add_edge(("a", a), ("a", int(b)), kind="mesh")
        # connectivity guard
        comps = list(nx.connected_components(Gr))
        while len(comps) > 1:
            u = next(iter(comps[0])); v = next(iter(comps[1])); Gr.add_edge(u, v, kind="mesh"); comps = list(nx.connected_components(Gr))
        gws = rng.choice(n_acc, n_srv, replace=False)
        for k, a in enumerate(gws):
            Gr.add_edge(("a", int(a)), ("s", k), kind="fiber")
        # links list (directed usage modelled as undirected capacity, uplinks separate)
        links = []; lid = {}
        for (u, v, dd) in Gr.edges(data=True):
            key = frozenset([u, v]); lid[key] = len(links); links.append(dd["kind"])
        up = {}
        dca = np.linalg.norm(cc[:, None] - acc[None], axis=2)
        for i in range(n_cam):
            for a in np.argsort(dca[i])[:2]:
                up[(i, int(a))] = len(links); links.append("up")
        self.links = np.array(links)
        self.link_region = np.full(len(links), -1)
        self.link_region2 = np.full(len(links), -1)
        for key, e in lid.items():
            ns = sorted(n[1] for n in key if n[0] == "a")
            self.link_region[e] = ns[0] if ns else -1
            self.link_region2[e] = ns[1] if len(ns) > 1 else -1
        for (i, a), e in up.items():
            self.link_region[e] = a
        nE = len(links)
        # capacities: load factor = all-full demand / total gateway capacity
        demand_full = n_cam * B[0]
        C_fiber = demand_full / (n_srv * load)
        C = np.where(self.links == "fiber", C_fiber, np.where(self.links == "mesh", 0.55 * C_fiber, 16.0 / load))
        self.C0 = C
        self.dly0 = np.where(self.links == "fiber", 1.0, np.where(self.links == "mesh", 3.0, 4.0))
        self.H0 = np.full(n_srv, n_cam * G[0] / (n_srv * gpu_load * load))
        # candidate routes per camera: shortest paths from each uplink AP to each server, keep P with path diversity
        routes = []   # (cam, server, [links])
        self.cam_routes = []
        for i in range(n_cam):
            cands = []
            for a in np.argsort(dca[i])[:2]:
                for k in range(n_srv):
                    pth = nx.shortest_path(Gr, ("a", int(a)), ("s", k))
                    le = [up[(i, int(a))]] + [lid[frozenset([pth[j], pth[j + 1]])] for j in range(len(pth) - 1)]
                    cands.append((len(le), int(a), k, le))
            cands.sort(key=lambda t: t[0])
            chosen = []; seen = set()
            for c in cands:                       # first pass: distinct (uplink, server) and diversity
                if (c[1], c[2]) in seen: continue
                if len(chosen) < P and (not chosen or c[1] != chosen[0][1] or c[2] != chosen[0][2]):
                    chosen.append(c); seen.add((c[1], c[2]))
            r = []
            for c in chosen[:P]:
                r.append(len(routes)); routes.append((i, c[2], c[3]))
            self.cam_routes.append(r)
        self.routes = routes; self.nE = nE
        self.A = np.zeros((nE, len(routes)))
        for p, (i, k, le) in enumerate(routes):
            self.A[le, p] = 1
        # decision index: (route p, mode m) -> camera i known from route
        self.dec = [(p, m) for p in range(len(routes)) for m in range(3)]
        self.dec_cam = np.array([routes[p][0] for p, m in self.dec])
        self.dec_srv = np.array([routes[p][1] for p, m in self.dec])
        self.dec_mode = np.array([m for p, m in self.dec])
        self.dec_route = np.array([p for p, m in self.dec])
        self.nd = len(self.dec)
        self.prio = np.array([max(self.w[V[i] > 0]) if (V[i] > 0).any() else 0 for i in range(n_cam)])

    # ---------------- scenarios ----------------
    def scenarios(self, n, rng, kappa=1.0, nominal=False):
        out = []
        for _ in range(n):
            mult = np.ones(self.nE); srv = np.ones(self.n_srv); comp = np.zeros(self.n_cam, bool)
            if not nominal:
                wl = self.links != "fiber"
                # spatially correlated shadowing per region + independent fading
                reg = np.exp(rng.normal(0, 0.25 * min(kappa, 2), self.n_acc))
                f = np.exp(rng.normal(0, 0.25, self.nE))
                rf = np.where(self.link_region >= 0, reg[np.clip(self.link_region, 0, None)], 1.0)
                mult = np.where(wl, np.clip(rf * f, 0.05, 1.0), 1.0)
                if rng.uniform() < min(0.9, 0.20 * kappa):            # regional outage of one access node
                    a = rng.integers(self.n_acc); mult[(self.link_region == a) | (self.link_region2 == a)] = 0.0
                fail = (rng.uniform(size=self.nE) < 0.03 * kappa) & (self.links == "mesh")
                mult[fail] = 0.0
                ov = rng.uniform(size=self.n_srv) < min(0.9, 0.15 * kappa)
                srv[ov] = rng.uniform(0.3, 0.7, ov.sum())
                comp = rng.uniform(size=self.n_cam) < 0.03 * kappa
            out.append(self._scen(mult, srv, comp))
        return out

    def _scen(self, mult, srv, comp):
        C = self.C0 * mult; H = self.H0 * srv
        with np.errstate(divide="ignore"):
            dl = np.where(mult > 0, self.dly0 / np.maximum(mult, 1e-9), np.inf)
        rdl = np.array([dl[le].sum() for (_, _, le) in self.routes])
        alive = np.array([np.all(mult[le] > 0) for (_, _, le) in self.routes])
        # q[z, d]: on-time detection contribution of decision d to zone z
        tot = rdl[self.dec_route] + PROC[self.dec_mode] / srv[self.dec_srv]
        q = self.V[self.dec_cam].T * RM[self.dec_mode][None, :]
        ontime = (tot[None, :] <= self.D[:, None]) & alive[self.dec_route][None, :]
        q = q * ontime
        return dict(C=C, H=H, q=q, comp=comp, mult=mult, srv=srv)


# ---------------- optimisation models ----------------
def _first_stage_blocks(I):
    """Rows: one selection per camera; nominal link and GPU shield."""
    nd = I.nd
    rows = []; lb = []; ub = []
    Sel = np.zeros((I.n_cam, nd)); Sel[I.dec_cam, np.arange(nd)] = 1
    Lk = I.A[:, I.dec_route] * B[I.dec_mode][None, :]
    Gk = np.zeros((I.n_srv, nd)); Gk[I.dec_srv, np.arange(nd)] = G[I.dec_mode]
    return Sel, Lk, Gk


def solve_two_stage(I, scens, eps=None, alpha=0.9, mu=0.02, objective="loss", time_limit=300, allowed_modes=(0, 1, 2), fixed_x=None, mip_gap=1e-3, surrogate="concave", routes_allowed=None, lam=1.0):
    """Two-stage scenario MILP. First stage x (binary, nonanticipative); recourse y_s (continuous frame fraction,
    downgrade on the same route only). If eps given, CVaR_alpha(L) <= eps. objective: 'loss' (expected loss) or 'cvar'."""
    nd = I.nd; S = len(scens); Z = I.n_zone; nE = I.nE; K = I.n_srv
    # variable layout: x[nd] | y[S*nd] | h[S*Z] | eta | xi[S]
    ox = 0; oy = nd; oh = oy + S * nd; oeta = oh + S * Z; oxi = oeta + 1; nv = oxi + S
    c = np.zeros(nv)
    totB = I.n_cam * B[0]; totG = I.n_cam * G[0]
    c[ox:ox + nd] = mu * (B[I.dec_mode] / totB + G[I.dec_mode] / totG)
    if objective == "loss":
        for s in range(S):
            c[oh + s * Z: oh + (s + 1) * Z] = -I.w / S
    elif objective == "mixed":  # expected loss + lam * CVaR
        for s in range(S):
            c[oh + s * Z: oh + (s + 1) * Z] = -I.w / S
        c[oeta] = lam; c[oxi:oxi + S] = lam / ((1 - alpha) * S)
    else:  # minimise CVaR
        c[oeta] = 1.0; c[oxi:oxi + S] = 1.0 / ((1 - alpha) * S)
    R, Cc, V, lo, hi = [], [], [], [], []
    r = [0]
    def add(cols, vals, l, u):
        R.extend([r[0]] * len(cols)); Cc.extend(cols); V.extend(vals); lo.append(l); hi.append(u); r[0] += 1
    Sel, Lk, Gk = _first_stage_blocks(I)
    for i in range(I.n_cam):
        cols = np.nonzero(Sel[i])[0]; add(list(ox + cols), [1] * len(cols), -np.inf, 1)
    for e in range(nE):
        cols = np.nonzero(Lk[e])[0]
        if len(cols): add(list(ox + cols), list(Lk[e, cols]), -np.inf, I.C0[e])
    for k in range(K):
        cols = np.nonzero(Gk[k])[0]; add(list(ox + cols), list(Gk[k, cols]), -np.inf, I.H0[k])
    # downgrade linkage: y[s,(p,m)] <= sum_{m' <= m (higher quality)} x[(p,m')] ; and per-route sum y <= sum x
    idx = {(p, m): d for d, (p, m) in enumerate(I.dec)}
    for s in range(S):
        sc = scens[s]
        for d in range(nd):        # recourse: frame shedding only, y <= x
            add([oy + s * nd + d, ox + d], [1, -1], -np.inf, 0)
        for e in range(nE):
            cols = np.nonzero(Lk[e])[0]
            if len(cols): add(list(oy + s * nd + cols), list(Lk[e, cols]), -np.inf, sc["C"][e])
        for k in range(K):
            cols = np.nonzero(Gk[k])[0]; add(list(oy + s * nd + cols), list(Gk[k, cols]), -np.inf, sc["H"][k])
        qq = sc["q"] * (~sc["comp"][I.dec_cam])[None, :]
        lam = KAPPA * -np.log(1 - np.clip(qq, 0, 0.999)) if surrogate == "concave" else qq
        for z in range(Z):
            cols = np.nonzero(lam[z])[0]
            if surrogate == "concave":
                for u0 in U0:
                    sl = np.exp(-u0)
                    add([oh + s * Z + z] + list(oy + s * nd + cols), [1] + list(-sl * lam[z, cols]), -np.inf, 1 - np.exp(-u0) - sl * u0)
            else:
                add([oh + s * Z + z] + list(oy + s * nd + cols), [1] + list(-lam[z, cols]), -np.inf, 0)
        if eps is not None or objective in ("cvar", "mixed"):
            # xi_s >= L_s - eta  ->  -xi_s - eta - sum w h  <= -1 (since L = 1 - sum w h)
            add([oxi + s, oeta] + list(oh + s * Z + np.arange(Z)), [-1, -1] + list(-I.w), -np.inf, -1)
    if eps is not None:
        add([oeta] + list(oxi + np.arange(S)), [1] + [1.0 / ((1 - alpha) * S)] * S, -np.inf, eps)
    Amat = coo_matrix((V, (R, Cc)), shape=(r[0], nv)).tocsr()
    lb = np.zeros(nv); ub = np.ones(nv); ub[oeta] = 1; ub[oxi:] = 1
    bad = ~np.isin(I.dec_mode, allowed_modes)
    if routes_allowed is not None: bad |= ~np.isin(I.dec_route, routes_allowed)
    ub[ox:ox + nd][bad] = 0
    if fixed_x is not None:
        lb[ox:ox + nd] = fixed_x; ub[ox:ox + nd] = fixed_x
    integ = np.zeros(nv); integ[ox:ox + nd] = 1
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(Amat, np.array(lo), np.array(hi)), integrality=integ,
               bounds=Bounds(lb, ub), options=dict(time_limit=time_limit, mip_rel_gap=mip_gap, disp=False))
    dt = time.time() - t0
    if res.x is None:
        return None, dict(status=res.status, time=dt)
    x = np.round(res.x[ox:ox + nd])
    hs = res.x[oh:oh + S * Z].reshape(S, Z)
    L = 1 - hs @ I.w
    cv = res.x[oeta] + res.x[oxi:oxi + S].sum() / ((1 - alpha) * S)
    return x, dict(status=res.status, time=dt, obj=res.fun, L=L, cvar_model=cv, gap=getattr(res, "mip_gap", np.nan))


def cvar(L, alpha):
    L = np.sort(np.asarray(L))[::-1]
    k = max(1, int(np.ceil((1 - alpha) * len(L))))
    return L[:k].mean()


# ---------------- baselines ----------------
def jrra_adapted(I, time_limit=60):
    """Priority-weighted rate utility, min acceptable rate (F or R only), joint route choice, nominal capacity."""
    nd = I.nd
    util = np.log1p(B[I.dec_mode]) / np.log1p(B[0])
    c = -(I.prio[I.dec_cam] * util)
    Sel, Lk, Gk = _first_stage_blocks(I)
    A = np.vstack([Sel, Lk, Gk]); hi = np.concatenate([np.ones(I.n_cam), I.C0, I.H0])
    ub = np.where(I.dec_mode <= 1, 1, 0)
    res = milp(c, constraints=LinearConstraint(A, -np.inf, hi), integrality=np.ones(nd), bounds=Bounds(0, ub),
               options=dict(time_limit=time_limit, disp=False))
    return np.round(res.x)


def lasp_fixed(I, mode=1):
    """Load-aware shortest path, fixed ROI mode, admission in priority order under nominal capacity."""
    x = np.zeros(I.nd); loadE = np.zeros(I.nE); loadK = np.zeros(I.n_srv)
    for i in np.argsort(-I.prio, kind="stable"):
        best = None
        for p in I.cam_routes[i]:
            le = I.A[:, p] > 0; k = I.routes[p][1]
            if np.all(loadE[le] + B[mode] <= I.C0[le]) and loadK[k] + G[mode] <= I.H0[k]:
                u = max(np.max((loadE[le] + B[mode]) / I.C0[le]), (loadK[k] + G[mode]) / I.H0[k])
                if best is None or u < best[0]: best = (u, p)
        if best:
            p = best[1]; d = p * 3 + mode; x[d] = 1
            loadE[I.A[:, p] > 0] += B[mode]; loadK[I.routes[p][1]] += G[mode]
    return x


def ras_greedy(I, scens, alpha=0.9, lam=1.0, mu=0.02, max_iter=None, surrogate="concave", mask=None, init=None):
    """Scalable online heuristic: greedy switching of (mode, route) per camera to minimise
    mean(L) + lam*CVaR(L) + resource cost on sampled scenarios, with a nominal capacity shield.
    Per-scenario survival of a stream is approximated by its route bottleneck capacity ratio."""
    S = len(scens); nd = I.nd
    surv = np.zeros((S, nd))
    Q = np.zeros((S, I.n_zone, nd))
    for s, sc in enumerate(scens):
        rat = np.where(I.C0 > 0, np.minimum(1, sc["C"] / I.C0), 0)
        rs = np.array([rat[le].min() for (_, _, le) in I.routes])
        surv[s] = rs[I.dec_route] * np.minimum(1, sc["H"] / I.H0)[I.dec_srv]
        qq = sc["q"] * surv[s][None, :] * (~sc["comp"][I.dec_cam])[None, :]
        Q[s] = KAPPA * -np.log(1 - np.clip(qq, 0, 0.999)) if surrogate == "concave" else qq
    totB = I.n_cam * B[0]; totG = I.n_cam * G[0]
    rc = mu * (B[I.dec_mode] / totB + G[I.dec_mode] / totG)
    cur = -np.ones(I.n_cam, int)      # current decision per camera (-1 = suspended)
    cov = np.zeros((S, I.n_zone))
    loadE = np.zeros(I.nE); loadK = np.zeros(I.n_srv)
    def J(cv):
        L = 1 - ((1 - np.exp(-cv)) if surrogate == "concave" else np.minimum(cv, 1)) @ I.w
        return L.mean() + lam * cvar(L, alpha)
    if init is not None:
        for d in np.nonzero(init)[0]:
            i = I.dec_cam[d]; cur[i] = d; cov += Q[:, :, d]
            loadE += I.A[:, I.dec_route[d]] * B[I.dec_mode[d]]; loadK[I.dec_srv[d]] += G[I.dec_mode[d]]
    base = J(cov); it = 0
    if mask is None: mask = np.ones(nd, bool)
    cam_dec = [np.nonzero((I.dec_cam == i) & mask)[0] for i in range(I.n_cam)]
    while True:
        it += 1
        best = (0.0, None)
        for i in range(I.n_cam):
            o = cur[i]
            covo = cov - (Q[:, :, o] if o >= 0 else 0)
            lE = loadE - (I.A[:, I.dec_route[o]] * B[I.dec_mode[o]] if o >= 0 else 0)
            lK = loadK.copy()
            if o >= 0: lK[I.dec_srv[o]] -= G[I.dec_mode[o]]
            for d in cam_dec[i]:
                if d == o: continue
                le = I.A[:, I.dec_route[d]] > 0
                if np.any(lE[le] + B[I.dec_mode[d]] > I.C0[le] + 1e-9): continue
                if lK[I.dec_srv[d]] + G[I.dec_mode[d]] > I.H0[I.dec_srv[d]] + 1e-9: continue
                val = J(covo + Q[:, :, d]) + rc[d] - (rc[o] if o >= 0 else 0) - base
                if val < best[0] - 1e-9: best = (val, (i, d))
        if best[1] is None or (max_iter and it > max_iter): break
        i, d = best[1]; o = cur[i]
        if o >= 0:
            cov -= Q[:, :, o]; loadE -= I.A[:, I.dec_route[o]] * B[I.dec_mode[o]]; loadK[I.dec_srv[o]] -= G[I.dec_mode[o]]
        cov += Q[:, :, d]; loadE += I.A[:, I.dec_route[d]] * B[I.dec_mode[d]]; loadK[I.dec_srv[d]] += G[I.dec_mode[d]]
        cur[i] = d; base = J(cov)
    x = np.zeros(nd); x[cur[cur >= 0]] = 1
    return x


# ---------------- evaluation ----------------
def recourse(I, x, sc):
    """Realised-scenario LP: shed/downgrade streams (same route) to respect realised capacities,
    maximising weighted coverage with nominal (non-compromise-aware) utilities."""
    nd = I.nd; Z = I.n_zone
    act = np.nonzero(x)[0]
    if len(act) == 0: return np.zeros(nd)
    cand = list(act)
    cand = np.array(cand)
    n = len(cand); nv = n + Z
    c = np.zeros(nv); c[n:] = -I.w
    c[:n] = 1e-4 * B[I.dec_mode[cand]]
    rows = []; hi = []
    Lk = I.A[:, I.dec_route[cand]] * B[I.dec_mode[cand]][None, :]
    rows.append(np.hstack([Lk, np.zeros((I.nE, Z))])); hi.append(sc["C"])
    Gk = np.zeros((I.n_srv, n)); Gk[I.dec_srv[cand], np.arange(n)] = G[I.dec_mode[cand]]
    rows.append(np.hstack([Gk, np.zeros((I.n_srv, Z))])); hi.append(sc["H"])
    # one active mode-fraction per route
    routes = np.unique(I.dec_route[cand])
    Rr = np.zeros((len(routes), nv))
    for j, p in enumerate(routes): Rr[j, :n] = (I.dec_route[cand] == p)
    rows.append(Rr); hi.append(np.ones(len(routes)))
    lam = KAPPA * -np.log(1 - np.clip(sc["q"][:, cand], 0, 0.999))
    for u0 in U0:
        sl = np.exp(-u0)
        rows.append(np.hstack([-sl * lam, np.eye(Z)])); hi.append(np.full(Z, 1 - np.exp(-u0) - sl * u0))
    res = linprog(c, A_ub=np.vstack(rows), b_ub=np.concatenate(hi), bounds=[(0, 1)] * nv, method="highs")
    y = np.zeros(nd); y[cand] = res.x[:n]
    return y


def detect_prob(qs):
    """P(at least one view detects) with Gaussian-copula correlated views (equicorrelation RHO)."""
    qs = qs[qs > 1e-9]
    if len(qs) == 0: return 0.0
    if len(qs) == 1: return float(qs[0])
    thr = norm.ppf(np.clip(qs, 1e-9, 1 - 1e-9))
    pm = norm.cdf((thr[None, :] - np.sqrt(RHO) * GH_X[:, None]) / np.sqrt(1 - RHO))
    return float(1 - (GH_W * np.prod(1 - pm, axis=1)).sum())


def evaluate(I, x, scens, ys=None):
    Ls, worst, bw, gpu, surro = [], [], [], [], []
    for j, sc in enumerate(scens):
        y = recourse(I, x, sc) if ys is None else ys[j]
        qe = sc["q"] * y[None, :] * (~sc["comp"][I.dec_cam])[None, :]
        pz = np.array([detect_prob(qe[z]) for z in range(I.n_zone)])
        Ls.append(1 - pz @ I.w); worst.append(pz[I.crit].min())
        surro.append(1 - (1 - np.exp(-KAPPA * -np.log(1 - np.clip(qe, 0, .999)).sum(1))) @ I.w)
        bw.append((B[I.dec_mode] * y).sum()); gpu.append((G[I.dec_mode] * y).sum())
    Ls = np.array(Ls)
    return dict(recall=1 - Ls.mean(), cvar95=cvar(Ls, 0.95), cvar90=cvar(Ls, 0.90), worst=np.mean(worst),
                bw=np.mean(bw), gpu=np.mean(gpu), prov_bw=(B[I.dec_mode] * x).sum() if x is not None else np.nan, surro=np.mean(surro), L=Ls)


def oracle(I, scens):
    ys = []
    for sc in scens:
        sc2 = dict(sc); sc2["comp"] = sc["comp"]
        x, info = solve_two_stage(I, [sc2], time_limit=30)
        y = recourse(I, x, dict(sc, q=sc["q"] * (~sc["comp"][I.dec_cam])[None, :]))
        ys.append(y)
    return ys


# ---------------- improved online controller (load-aware survival, ratio rule, best-of-two) ----------------
def _prep(I, scens, surrogate):
    S = len(scens)
    Cs = np.array([sc["C"] for sc in scens])            # S x E
    Hs = np.array([sc["H"] for sc in scens])            # S x K
    L0 = np.zeros((S, I.n_zone, I.nd))
    for s, sc in enumerate(scens):
        qq = sc["q"] * (~sc["comp"][I.dec_cam])[None, :]
        L0[s] = KAPPA * -np.log(1 - np.clip(qq, 0, 0.999)) if surrogate == "concave" else qq
    route_links = [np.nonzero(I.A[:, p])[0] for p in range(len(I.routes))]
    return Cs, Hs, L0, route_links


def ras_greedy2(I, scens, alpha=0.95, lam=1.0, mu=0.02, surrogate="concave", mask=None, init=None,
                rule="gain", max_iter=10000, max_prov_bw=None, max_prov_gpu=None):
    """Greedy switching with load-aware proportional-shedding survival.
    Survival of stream d in scenario s: min over its route links of min(1, C_es / load_e) times
    min(1, H_ks / load_k), where loads include the candidate. Survival of the other streams is
    refreshed after every accepted move (fixed-point approximation).
    rule='gain' picks the largest objective decrease, rule='ratio' the largest decrease per unit of
    normalised resource increase."""
    S = len(scens); nd = I.nd
    Cs, Hs, L0, rl = _prep(I, scens, surrogate)
    totB = I.n_cam * B[0]; totG = I.n_cam * G[0]
    rc = mu * (B[I.dec_mode] / totB + G[I.dec_mode] / totG)
    res_norm = B[I.dec_mode] / totB + G[I.dec_mode] / totG
    if mask is None: mask = np.ones(nd, bool)
    cam_dec = [np.nonzero((I.dec_cam == i) & mask)[0] for i in range(I.n_cam)]
    cur = -np.ones(I.n_cam, int)
    loadE = np.zeros(I.nE); loadK = np.zeros(I.n_srv)
    def cover_fn(u):
        return (1 - np.exp(-u)) if surrogate == "concave" else np.minimum(u, 1)
    def J_of(u):
        L = 1 - cover_fn(u) @ I.w
        return L.mean() + lam * cvar(L, alpha)
    def survival(d, lE, lK):
        le = rl[I.dec_route[d]]
        sh = np.min(np.minimum(1, Cs[:, le] / np.maximum(lE[le], 1e-9)[None, :]), axis=1)
        k = I.dec_srv[d]
        return sh * np.minimum(1, Hs[:, k] / max(lK[k], 1e-9))
    def contrib(d, lE, lK):
        return L0[:, :, d] * survival(d, lE, lK)[:, None]
    def full_u(cur, lE, lK):
        u = np.zeros((S, I.n_zone))
        for d in cur[cur >= 0]: u += contrib(d, lE, lK)
        return u
    def add_load(d, lE, lK, sgn):
        lE[rl[I.dec_route[d]]] += sgn * B[I.dec_mode[d]]; lK[I.dec_srv[d]] += sgn * G[I.dec_mode[d]]
    if init is not None:
        for d in np.nonzero(init)[0]:
            cur[I.dec_cam[d]] = d; add_load(d, loadE, loadK, +1)
    def _prov(cur):
        ds = cur[cur >= 0]
        if len(ds) == 0: return 0.0, 0.0
        return float(B[I.dec_mode[ds]].sum()), float(G[I.dec_mode[ds]].sum())
    while True:
        pb, pg = _prov(cur)
        ok = True
        if max_prov_bw is not None and pb > max_prov_bw + 1e-9: ok = False
        if max_prov_gpu is not None and pg > max_prov_gpu + 1e-9: ok = False
        if ok or not (cur >= 0).any(): break
        act = [d for d in cur if d >= 0]
        d_drop = min(act, key=lambda d: I.prio[I.dec_cam[d]])
        add_load(d_drop, loadE, loadK, -1); cur[I.dec_cam[d_drop]] = -1
    u = full_u(cur, loadE, loadK)
    base = J_of(u) + rc[cur[cur >= 0]].sum()
    it = 0
    while it < max_iter:
        it += 1
        cand = []
        for i in range(I.n_cam):
            o = cur[i]
            lE = loadE.copy(); lK = loadK.copy()
            u_o = u.copy()
            if o >= 0:
                u_o -= contrib(o, loadE, loadK); add_load(o, lE, lK, -1)
            if o >= 0:                          # suspension move
                val = J_of(u_o) + rc[cur[cur >= 0]].sum() - rc[o] - base
                if val < -1e-9:
                    cand.append((val if rule == "gain" else val / 1e-3, i, -1))
            for d in cam_dec[i]:
                if d == o: continue
                le = rl[I.dec_route[d]]
                if np.any(lE[le] + B[I.dec_mode[d]] > I.C0[le] + 1e-9): continue
                k = I.dec_srv[d]
                if lK[k] + G[I.dec_mode[d]] > I.H0[k] + 1e-9: continue
                others = [cur[j] for j in range(I.n_cam) if j != i and cur[j] >= 0]
                pb = B[I.dec_mode[d]] + (B[I.dec_mode[others]].sum() if others else 0)
                pg = G[I.dec_mode[d]] + (G[I.dec_mode[others]].sum() if others else 0)
                if max_prov_bw is not None and pb > max_prov_bw + 1e-9: continue
                if max_prov_gpu is not None and pg > max_prov_gpu + 1e-9: continue
                lE2 = lE.copy(); lK2 = lK.copy(); add_load(d, lE2, lK2, +1)
                val = J_of(u_o + contrib(d, lE2, lK2)) + rc[cur[cur >= 0]].sum() - (rc[o] if o >= 0 else 0) + rc[d] - base
                if val >= -1e-9: continue
                if rule == "ratio":
                    dr = res_norm[d] - (res_norm[o] if o >= 0 else 0)
                    score = val / max(dr, 1e-3)
                else:
                    score = val
                cand.append((score, i, d))
        if not cand: break
        cand.sort(key=lambda t: t[0])
        moved = False
        for _, i, d in cand[:10]:              # verify with refreshed survival of all streams
            o = cur[i]; lE = loadE.copy(); lK = loadK.copy(); c2 = cur.copy()
            if o >= 0: add_load(o, lE, lK, -1)
            if d >= 0: add_load(d, lE, lK, +1)
            c2[i] = d
            u2 = full_u(c2, lE, lK); b2 = J_of(u2) + rc[c2[c2 >= 0]].sum()
            if b2 < base - 1e-9:
                cur, loadE, loadK, u, base = c2, lE, lK, u2, b2; moved = True; break
        if not moved: break
    x = np.zeros(nd); x[cur[cur >= 0]] = 1
    return x, base


def ras_controller(I, scens, alpha=0.95, lam=1.0, **kw):
    """Proposed RAS-G controller: risk-neutral warm start, then best of the gain rule and the
    ratio rule for the risk-aware objective (sampled objective decides)."""
    x0g, v0g = ras_greedy2(I, scens, alpha=alpha, lam=0.0, rule="gain", **kw)
    x0r, v0r = ras_greedy2(I, scens, alpha=alpha, lam=0.0, rule="ratio", **kw)
    if lam == 0:
        return x0g if v0g <= v0r else x0r
    cands = []
    for x0 in (x0g, x0r):
        for rule in ("gain", "ratio"):
            x, val = ras_greedy2(I, scens, alpha=alpha, lam=lam, rule=rule, init=x0, **kw)
            cands.append((val, x))
    return min(cands, key=lambda t: t[0])[1]
