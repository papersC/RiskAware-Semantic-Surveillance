"""Regenerate all result figures (pgfplots) and tables of the manuscript from the JSON results.

Usage: python3 make_figures.py   (reads ../data/v2_*.json, writes ../figures/*.tex, ../tables/*.tex and ../data/stats.json)
"""
import json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

M = ["RAS-G", "RN-G", "NOM-G", "ADD-G", "JRRA-A", "LASP-F"]
STY = {"RAS-G": "blue!75!black,thick,mark=*", "RN-G": "cyan!60!black,mark=o", "NOM-G": "red!75!black,mark=square*",
       "ADD-G": "teal!70!black,mark=triangle*", "JRRA-A": "orange!85!black,mark=diamond*",
       "LASP-F": "violet!80!black,mark=pentagon*", "ORACLE": "black,densely dashed",
       "RAS-G(matched)": "blue!75!black,densely dotted,mark=*,mark options={solid,fill=white}"}


def load(kind):
    return json.load(open(os.path.join(DATA, f"v2_{kind}.json")))


def ci(a):
    a = np.asarray(a, float)
    if len(a) < 2: return a.mean(), 0.0
    return a.mean(), stats.t.ppf(.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))


def col(R, name, metric, val=None):
    return np.array([r["res"][name][metric] for r in R if val is None or r["val"] == val])


def paired(R, a, b, metric):
    x, y = col(R, a, metric), col(R, b, metric)
    d = x - y
    return dict(mean=float(d.mean()), ci=float(ci(d)[1]), t_p=float(stats.ttest_rel(x, y).pvalue),
                w_p=float(stats.wilcoxon(x, y).pvalue) if np.any(d != 0) else 1.0, wins=int((d < 0).sum()), n=len(d))


def holm(pvals):
    keys = sorted(pvals, key=pvals.get); m = len(keys); out = {}; run = 0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k])); out[k] = run
    return out


def axis_open(opts):
    return "\\begin{tikzpicture}\n\\begin{axis}[" + opts + "]\n"


S = {}  # numbers for the text

# ---------------- main table ----------------
R = load("main")
rows = []
for n in M:
    rows.append((n, *[ci(col(R, n, m)) for m in ("recall", "cvar95", "worst")], col(R, n, "prov_bw").mean(), col(R, n, "time").mean()))
best = {m: max if m != "cvar95" else min for m in ("recall", "cvar95", "worst")}
tab = ["\\begin{table*}[t]", "\\caption{Results at $\\rho=1$ (mean $\\pm$ 95\\% CI, 15 topologies)}", "\\label{tab:main}",
       "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{8pt}", "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
       "Method & Recall (M1) & CVaR$_{0.95}$ (M2) & Worst zone (M3) & Mb/s (M4) & Time (s)\\\\ \\midrule"]
bvals = {m: (max if m != 1 else min)([r[m + 1][0] for r in rows]) for m in range(3)}
for r in rows:
    cells = []
    for j in range(3):
        mval, h = r[j + 1]
        s = f"{mval:.3f}"
        if abs(mval - bvals[j]) < 1e-12: s = "\\textbf{" + s + "}"
        cells.append(f"{s}$\\pm${h:.3f}")
    tab.append(f"{r[0]} & " + " & ".join(cells) + f" & {r[4]:.1f} & {r[5]:.2f} \\\\")
tab.append("\\midrule")
for n, lab in [("RAS-G@100", "RAS-G$^\\dagger$"), ("RN-G@100", "RN-G$^\\dagger$"), ("JRRA-A@100", "JRRA-A$^\\dagger$"), ("ORACLE@100", "ORACLE$^\\dagger$")]:
    vals = [ci(col(R, n, m)) for m in ("recall", "cvar95", "worst")]
    bw = "n/a" if n.startswith("ORACLE") else f"{col(R, n, 'prov_bw').mean():.1f}"
    tab.append(f"{lab} & " + " & ".join(f"{v:.3f}$\\pm${h:.3f}" for v, h in vals) + f" & {bw} & n/a \\\\")
tab += ["\\bottomrule", "\\multicolumn{6}{@{}l}{\\scriptsize $^\\dagger$First 100 test scenarios per topology. M4 is provisioned bandwidth.}",
        "\\end{tabular}", "\\end{table*}"]
open(os.path.join(TAB, "tab_main.tex"), "w").write("\n".join(tab) + "\n")
S["main"] = {n: {m: ci(col(R, n, m))[0] for m in ("recall", "cvar95", "worst", "prov_bw", "bw", "time")} for n in M}
S["main_100"] = {n: {m: ci(col(R, n, m))[0] for m in ("recall", "cvar95", "worst")} for n in ["RAS-G@100", "RN-G@100", "JRRA-A@100", "ORACLE@100"]}
pv = {b: paired(R, "RAS-G", b, "cvar95") for b in M[1:]}
hp = holm({b: v["w_p"] for b, v in pv.items()})
for b in pv: pv[b]["w_p_holm"] = hp[b]
S["paired_cvar"] = pv
S["paired_recall"] = {b: paired(R, "RAS-G", b, "recall") for b in M[1:]}
S["paired_worst"] = {b: paired(R, "RAS-G", b, "worst") for b in M[1:]}

# ---------------- tail CCDF ----------------
pool = {n: np.concatenate([r["tails"][n] for r in R]) for n in M + ["ORACLE@100"]}
xs = np.linspace(0, 0.8, 33)
s = [axis_open("width=0.95\\columnwidth,height=5.0cm,xlabel={Weighted missed-event loss $\\ell$},ylabel={$\\Pr(L>\\ell)$},ymode=log,"
              "ymin=1e-3,ymax=1,xmin=0,xmax=0.8,grid=major,grid style={gray!25},tick label style={font=\\scriptsize},"
              "label style={font=\\footnotesize},legend style={font=\\scriptsize,at={(0.99,0.99)},anchor=north east,legend columns=2},"
              "every axis plot/.append style={mark size=1.3pt,mark repeat=4}")]
for n in ["RAS-G", "RN-G", "NOM-G", "JRRA-A", "LASP-F", "ORACLE@100"]:
    L = pool[n]; pts = [(x, max((L > x).mean(), 1e-4)) for x in xs]
    pts = [(x, y) for x, y in pts if y > 1e-3]
    st = STY["ORACLE" if n.startswith("ORACLE") else n]
    s.append(f"\\addplot[{st}] coordinates {{" + "".join(f"({x:.3f},{y:.5f})" for x, y in pts) + "};")
    s.append(f"\\addlegendentry{{{'ORACLE$^\\dagger$' if n.startswith('ORACLE') else n}}}")
s += ["\\end{axis}", "\\end{tikzpicture}"]
open(os.path.join(FIG, "fig_tail_body.tex"), "w").write("\n".join(s) + "\n")
S["tail"] = {n: {"p_gt_0.3": float((pool[n] > 0.3).mean()), "p_gt_0.4": float((pool[n] > 0.4).mean()), "q99": float(np.quantile(pool[n], 0.99))} for n in pool}

# ---------------- ablation ----------------
R = load("abl")
base = {m: col(R, "RAS-G", m) for m in ("recall", "cvar95", "worst", "prov_bw")}
lab = {"RN-G": "No CVaR term ($\\lambda=0$)", "ADD-G": "Additive surrogate", "NOROUTE": "Shortest route only",
       "FULLONLY": "Full mode only", "NOM-G": "Nominal scenario only", "STATIC": "Static survival ratio",
       "GAINONLY": "Gain rule only"}
tab = ["\\begin{table}[t]", "\\caption{Ablation at $\\rho=1$ (change relative to RAS-G)}", "\\label{tab:ablation}", "\\centering\\footnotesize",
       "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}lcccc@{}}", "\\toprule", "Variant & $\\Delta$M1 & $\\Delta$M2 & $\\Delta$M3 & $\\Delta$M4 (Mb/s)\\\\ \\midrule"]
S["abl"] = {}
for n in ["RN-G", "ADD-G", "NOROUTE", "FULLONLY", "NOM-G", "STATIC", "GAINONLY"]:
    d = {m: col(R, n, m) - base[m] for m in base}
    p = stats.wilcoxon(col(R, n, "cvar95"), base["cvar95"]).pvalue if np.any(d["cvar95"] != 0) else 1.0
    S["abl"][n] = {m: float(d[m].mean()) for m in d}; S["abl"][n]["p_cvar"] = float(p)
    star = "$^{*}$" if p < 0.05 else ""
    z = lambda v, k: round(float(v), k) + 0.0
    tab.append(f"{lab[n]} & ${z(d['recall'].mean(),3):+.3f}$ & ${z(d['cvar95'].mean(),3):+.3f}${star} & ${z(d['worst'].mean(),3):+.3f}$ & ${z(d['prov_bw'].mean(),1):+.1f}$ \\\\")
tab += ["\\bottomrule", "\\multicolumn{5}{@{}l}{\\scriptsize $^{*}$Wilcoxon signed-rank $p<0.05$ for $\\Delta$M2 over 15 topologies.}", "\\end{tabular}", "\\end{table}"]
open(os.path.join(TAB, "tab_ablation.tex"), "w").write("\n".join(tab) + "\n")
S["abl_base"] = {m: float(v.mean()) for m, v in base.items()}

# ---------------- Pareto ----------------
R = load("pareto")
lams = [0, 0.25, 0.5, 1, 2, 4, 8]
P = [(col(R, f"lam{l}", "prov_bw").mean(), col(R, f"lam{l}", "cvar95").mean(), col(R, f"lam{l}", "recall").mean()) for l in lams]
S["pareto"] = {str(l): {"bw": p[0], "cvar": p[1], "recall": p[2]} for l, p in zip(lams, P)}
Rm = load("main")
others = {n: (col(Rm, n, "prov_bw").mean(), col(Rm, n, "cvar95").mean()) for n in ["NOM-G", "ADD-G", "JRRA-A", "LASP-F"]}
allx = [p[0] for p in P] + [v[0] for v in others.values()]; ally = [p[1] for p in P] + [v[1] for v in others.values()]
s = [axis_open(f"width=\\columnwidth,height=4.9cm,xlabel={{Provisioned bandwidth (Mb/s)}},ylabel={{CVaR$_{{0.95}}$ of loss}},grid=major,"
              f"grid style={{gray!25}},xmin={min(allx) - 8:.0f},xmax={max(allx) + 8:.0f},ymin={min(ally) - 0.02:.2f},ymax={max(ally) + 0.05:.2f},"
              "tick label style={font=\\scriptsize},label style={font=\\footnotesize},legend style={font=\\scriptsize,at={(0.99,0.99)},anchor=north east,legend columns=2}")]
s.append("\\addplot[blue!75!black,thick,mark=*,mark size=1.6pt] coordinates {" + "".join(f"({x:.2f},{y:.4f})" for x, y, _ in P) + "};")
s.append("\\addlegendentry{RAS-G ($\\lambda$ sweep)}")
mk = {"NOM-G": "red!75!black,mark=square*", "ADD-G": "teal!70!black,mark=triangle*", "JRRA-A": "orange!85!black,mark=diamond*", "LASP-F": "violet!80!black,mark=pentagon*"}
for n, (x, y) in others.items():
    s.append(f"\\addplot[only marks,{mk[n]},mark size=2.4pt] coordinates {{({x:.2f},{y:.4f})}};\\addlegendentry{{{n}}}")
anc = {0: "south", 0.5: "north", 8: "west"}
for l, (x, y, _) in zip(lams, P):
    if l in anc:
        s.append(f"\\node[font=\\tiny,anchor={anc[l]},inner sep=2pt] at (axis cs:{x:.2f},{y:.4f}) {{$\\lambda{{=}}{l}$}};")
s += ["\\end{axis}", "\\end{tikzpicture}"]
open(os.path.join(FIG, "fig_pareto_body.tex"), "w").write("\n".join(s) + "\n")


# ---------------- load / severity ----------------
def series(kind, metric):
    R = load(kind); vals = sorted(set(r["val"] for r in R))
    return {n: [(v, col(R, n, metric, v).mean()) for v in vals] for n in M}, vals


panels = [("load", "cvar95", "Load factor $\\rho$", "CVaR$_{0.95}$", "(a)"), ("load", "recall", "Load factor $\\rho$", "Recall (M1)", "(b)"),
          ("sev", "cvar95", "Severity $\\kappa_d$", "CVaR$_{0.95}$", "(c)"), ("sev", "worst", "Severity $\\kappa_d$", "Worst zone (M3)", "(d)")]
s = ["\\begin{tikzpicture}", "\\begin{groupplot}[group style={group size=4 by 1,horizontal sep=1.2cm},width=0.27\\textwidth,height=4.3cm,grid=major,grid style={gray!25},",
     " tick label style={font=\\scriptsize},label style={font=\\footnotesize},every axis plot/.append style={mark size=1.5pt},legend columns=6,legend style={font=\\scriptsize,draw=none,column sep=6pt}]"]
S["load"] = {}; S["sev"] = {}
for i, (k, m, xl, yl, tag) in enumerate(panels):
    opt = f"xlabel={{{xl}}},ylabel={{{yl}}},title={{\\footnotesize {tag}}}"
    if i == 0: opt += ",legend to name=lsleg"
    s.append(f"\\nextgroupplot[{opt}]")
    ser, vals = series(k, m)
    S[k][m] = {n: {str(v): y for v, y in pts} for n, pts in ser.items()}
    for n, pts in ser.items():
        s.append(f"\\addplot[{STY[n]}] coordinates {{" + "".join(f"({v},{y:.4f})" for v, y in pts) + "};")
        if i == 0: s.append(f"\\addlegendentry{{{n}}}")
s += ["\\end{groupplot}", "\\end{tikzpicture}", "\\\\[1mm]\\ref{lsleg}"]
open(os.path.join(FIG, "fig_loadsev_body.tex"), "w").write("\n".join(s) + "\n")
for k in ("load", "sev"):
    R = load(k); vals = sorted(set(r["val"] for r in R))
    S[k]["paired_rn"] = {}
    for v in vals:
        x, y = col(R, "RAS-G", "cvar95", v), col(R, "RN-G", "cvar95", v)
        S[k]["paired_rn"][str(v)] = {"d": float((x - y).mean()), "w_p": float(stats.wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0,
                                     "d_recall": float((col(R, "RAS-G", "recall", v) - col(R, "RN-G", "recall", v)).mean())}
    S[k]["prov_bw"] = {n: {str(v): float(col(R, n, "prov_bw", v).mean()) for v in vals} for n in M}
    S[k]["recall"] = {n: {str(v): float(col(R, n, "recall", v).mean()) for v in vals} for n in M}

# ---------------- robustness (rho mismatch, distribution shift, sample size) ----------------
Rr = load("rho"); Rs = load("shift"); Rz = load("ssize")
s = ["\\begin{tikzpicture}", "\\begin{groupplot}[group style={group size=3 by 1,horizontal sep=1.25cm},width=0.34\\textwidth,height=4.3cm,grid=major,grid style={gray!25},",
     " tick label style={font=\\scriptsize},label style={font=\\footnotesize},every axis plot/.append style={mark size=1.5pt},legend columns=7,legend style={font=\\scriptsize,draw=none,column sep=6pt}]"]
rhos = [0.0, 0.2, 0.4, 0.6, 0.8]
s.append("\\nextgroupplot[xlabel={Evaluation correlation},ylabel={CVaR$_{0.95}$},title={\\footnotesize (a)},legend to name=robleg]")
S["rho"] = {}
for n in ["RAS-G", "RN-G", "ADD-G", "JRRA-A", "LASP-F"]:
    pts = [(r_, col(Rr, f"{n}@{r_}", "cvar95").mean()) for r_ in rhos]
    S["rho"][n] = {str(a): b for a, b in pts}
    S["rho"][n + "_recall"] = {str(r_): col(Rr, f"{n}@{r_}", "recall").mean() for r_ in rhos}
    s.append(f"\\addplot[{STY[n]}] coordinates {{" + "".join(f"({a},{b:.4f})" for a, b in pts) + "};")
    s.append(f"\\addlegendentry{{{n}}}")
s.append("\\addlegendimage{" + STY["RAS-G(matched)"] + "}\\addlegendentry{RAS-G (matched)}")
kts = [1.0, 1.5, 2.0, 2.5]
s.append("\\nextgroupplot[xlabel={Test severity $\\kappa_d$ (trained at 1)},ylabel={CVaR$_{0.95}$},title={\\footnotesize (b)}]")
S["shift"] = {}
for n in ["RAS-G", "RN-G", "JRRA-A", "LASP-F", "RAS-G(matched)"]:
    pts = [(k_, col(Rs, f"{n}@{k_}", "cvar95").mean()) for k_ in kts]
    S["shift"][n] = {str(a): b for a, b in pts}
    s.append(f"\\addplot[{STY[n]}] coordinates {{" + "".join(f"({a},{b:.4f})" for a, b in pts) + "};")
S["shift_p"] = {}
for k_ in kts:
    x, y = col(Rs, f"RAS-G@{k_}", "cvar95"), col(Rs, f"RN-G@{k_}", "cvar95")
    S["shift_p"][str(k_)] = float(stats.wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0
Ss = [25, 50, 100, 200, 300, 500]
s.append("\\nextgroupplot[xlabel={Training scenarios $|\\hat{\\mathcal S}|$},ylabel={CVaR$_{0.95}$},xmode=log,log basis x=10,title={\\footnotesize (c)},"
         "xtick={25,50,100,200,500},xticklabels={25,50,100,200,500}]")
pts = [(z, ci(col(Rz, f"S{z}", "cvar95"))) for z in Ss]
S["ssize"] = {str(z): {"cvar": c[0], "ci": c[1], "time": col(Rz, f"S{z}", "time").mean(), "recall": col(Rz, f"S{z}", "recall").mean()} for z, c in pts}
s.append(f"\\addplot[{STY['RAS-G']},error bars/.cd,y dir=both,y explicit] coordinates {{" + "".join(f"({z},{c[0]:.4f}) +- (0,{c[1]:.4f})" for z, c in pts) + "};")
s += ["\\end{groupplot}", "\\end{tikzpicture}", "\\\\[1mm]\\ref{robleg}"]
open(os.path.join(FIG, "fig_robust_body.tex"), "w").write("\n".join(s) + "\n")

# ---------------- large networks ----------------
try:
    R = load("large")
    S["large"] = {n: {m: ci(col(R, n, m))[0] for m in ("recall", "cvar95", "worst", "prov_bw", "time")} for n in M}
    S["large_paired"] = {b: paired(R, "RAS-G", b, "cvar95") for b in M[1:]}
except FileNotFoundError:
    pass

# ---------------- scalability / gap ----------------
try:
    Sc = json.load(open(os.path.join(DATA, "v2_scale.json")))
    tab = ["\\begin{table}[t]", "\\caption{Optimality gap and runtime}", "\\label{tab:scale}", "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{3.2pt}",
           "\\begin{tabular}{@{}rccccc@{}}", "\\toprule",
           "Cameras & \\multicolumn{2}{c}{Gap RAS-G} & Gap static & MILP (s) & RAS-G (s)\\\\",
           " & $\\lambda=0$ & $\\lambda=1$ & $\\lambda=1$ & $\\lambda=1$ & $\\lambda=1$\\\\ \\midrule"]
    S["gap"] = {}
    for n in (8, 12, 16):
        g0 = [r for r in Sc["gap"] if r["ncam"] == n and r["lam"] == 0.0]; g1 = [r for r in Sc["gap"] if r["ncam"] == n and r["lam"] == 1.0]
        a0 = np.mean([r["obj_g"] - r["obj_m"] for r in g0]); a1 = np.mean([r["obj_g"] - r["obj_m"] for r in g1])
        st = np.mean([r["obj_static"] - r["obj_m"] for r in g1])
        tm = np.mean([r["t_m"] for r in g1]); tg = np.mean([r["t_g"] for r in g1])
        S["gap"][str(n)] = dict(g0=a0, g1=a1, static=st, tm=tm, tg=tg, max_g1=max(r["obj_g"] - r["obj_m"] for r in g1),
                                optimal=all(r["status"] == 0 for r in g0 + g1))
        tab.append(f"{n} & {a0:.4f} & {a1:.4f} & {st:.4f} & {tm:.1f} & {tg:.2f} \\\\")
    tab.append("\\midrule")
    S["scale"] = {}
    for n in (24, 48, 72, 96):
        rr = [r for r in Sc["scale"] if r["ncam"] == n]
        tg = np.mean([r["t_g"] for r in rr]); tl = np.mean([r["t_rec"] for r in rr])
        S["scale"][str(n)] = dict(tg=tg, trec=tl)
        tab.append(f"{n} & n/a & n/a & n/a & n/a & {tg:.2f} \\\\")
    tab += ["\\bottomrule", "\\multicolumn{6}{@{}l}{\\scriptsize Mean absolute gaps in loss units over 5 topologies with 60 scenarios.}\\\\", "\\multicolumn{6}{@{}l}{\\scriptsize Runtimes above 16 cameras use 300 scenarios and 3 topologies.}", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(TAB, "tab_scale.tex"), "w").write("\n".join(tab) + "\n")
except FileNotFoundError:
    pass

json.dump(S, open(os.path.join(DATA, "stats.json"), "w"), indent=1, default=float)
print("figures and tables written")
