"""
analysis.py  -  label-free fingerprint analysis for the ICPRS paper.

Run from the project folder (pfe_thesis):
    .\\venv\\Scripts\\python.exe analysis.py

Reads the saved results (no model, no GPU, no API needed) and writes
everything to the folder  analysis_out/ :
    summary.txt            <- all numbers (send this file back)
    fig_risk_coverage.png/.pdf
    fig_fingerprint_regions.png/.pdf

Key idea: regions are defined from SC and SE only (medians, no labels).
Correctness labels are used only AFTER, to measure error rates.
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2

import json
import sys
JUDGE = "--judge-labels" in sys.argv     # use Claude-judge labels instead of containment matching
OUT = "analysis_out_judge" if JUDGE else "analysis_out"
os.makedirs(OUT, exist_ok=True)
LABELS = None
if JUDGE:
    with open(os.path.join("analysis_out", "judge_labels.json"), encoding="utf-8") as _f:
        LABELS = json.load(_f)


def lab(name, r):
    """True if the answer counts as correct under the chosen labelling."""
    if LABELS is None:
        return r["is_correct"]
    default = "CORRECT" if r["is_correct"] else "INCORRECT"
    return LABELS.get(f"{name}#{r['idx']}", default) == "CORRECT"
rng = np.random.default_rng(42)
log = open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")
    log.flush()


SOURCES = {
    "Mistral-7B | TriviaQA": os.path.join("results", "final_results.pkl"),
    "Qwen2-7B | TriviaQA": os.path.join("results_llama3", "final_results.pkl"),
    "Qwen2-1.5B | TriviaQA": os.path.join("results_phi3", "final_results.pkl"),
    "Mistral-7B | NaturalQA": os.path.join("results_naturalqa", "final_results.pkl"),
    "Mistral-7B | HotpotQA": os.path.join("results_hotpotqa", "final_results.pkl"),
}
TRIVIA = list(SOURCES)[:3]
B_BOOT = 1000     # bootstrap resamples for AUROC intervals
REPS = 500        # bootstrap + calibration/evaluation split repetitions


# ---------------------------------------------------------------- loading
def load(path, name):
    with open(path, "rb") as f:
        d = pickle.load(f)
    res = d["results"] if isinstance(d, dict) else d

    def col(k):
        if k not in res[0]:
            return None
        return np.array([r[k] for r in res], dtype=float)

    return {
        "wrong": np.array([0 if lab(name, r) else 1 for r in res]),
        "sc": col("uncertainty_sc"),
        "se": col("uncertainty_se"),
        "lp": col("avg_neg_log_prob"),
        "nli": col("uncertainty_selfcheck"),
    }


def comb(arrs, ref):
    """Equal-weight mean of min-max normalised signals (ranges from `ref` only)."""
    out = []
    for a in arrs:
        lo, hi = a[ref].min(), a[ref].max()
        out.append(np.clip((a - lo) / (hi - lo + 1e-12), 0, 1))
    return np.mean(out, axis=0)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


# ------------------------------------------------------- 1. AUROC + CIs
def table_auc(d):
    y, n = d["wrong"], len(d["wrong"])
    ref = np.arange(n)
    sc = {"Avg(-logP)": d["lp"], "Self-Consistency": d["sc"],
          "Semantic Entropy": d["se"]}
    if d["nli"] is not None:
        sc["SelfCheckGPT-NLI"] = d["nli"]
    sc["Combined (SC+SE+LP)"] = comb([d["sc"], d["se"], d["lp"]], ref)
    if d["nli"] is not None:
        sc["Combined (SC+SE+LP+NLI)"] = comb(
            [d["sc"], d["se"], d["lp"], d["nli"]], ref)
    vals = {k: [] for k in sc}
    for _ in range(B_BOOT):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        for k, s in sc.items():
            vals[k].append(roc_auc_score(y[i], s[i]))
    P("  [AUROC for detecting wrong answers, 95% bootstrap CI]")
    for k, s in sc.items():
        lo, hi = np.percentile(vals[k], [2.5, 97.5])
        P(f"    {k:26s} {roc_auc_score(y, s):.4f}  [{lo:.3f}, {hi:.3f}]")
    if "SelfCheckGPT-NLI" in sc:
        for k in [x for x in sc if x.startswith("Combined")]:
            diff = np.array(vals[k]) - np.array(vals["SelfCheckGPT-NLI"])
            lo, hi = np.percentile(diff, [2.5, 97.5])
            P(f"    paired diff {k} - SelfCheck-NLI: "
              f"{diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"
              f"   (CI containing 0 = no clear difference)")


# ----------------------------------------------- 2. label-free regions
def region_masks(sc, se, t_sc, t_se):
    hsc, hse = sc > t_sc, se > t_se
    return {
        "Confident (low SC, low SE)": ~hsc & ~hse,
        "Gap-like (high SC, high SE)": hsc & hse,
        "Shallow (low SC, high SE)": ~hsc & hse,
        "Surface-varying (high SC, low SE)": hsc & ~hse,
    }


def regions_report(d):
    y, sc, se = d["wrong"], d["sc"], d["se"]
    n = len(y)
    t_sc, t_se = np.median(sc), np.median(se)
    P(f"  [Regions: thresholds = medians, SC>{t_sc:.3f}, SE>{t_se:.3f}; "
      f"no labels used]")
    P(f"    {'region':36s} {'n':>4s} {'%':>6s} {'errors':>6s}  error rate [95% CI]")
    for name, m in region_masks(sc, se, t_sc, t_se).items():
        k, nn = int(y[m].sum()), int(m.sum())
        lo, hi = wilson(k, nn)
        rate = k / nn if nn else float("nan")
        P(f"    {name:36s} {nn:4d} {100*nn/n:5.1f}% {k:6d}  "
          f"{rate:.3f} [{lo:.3f}, {hi:.3f}]")
    m = region_masks(sc, se, t_sc, t_se)
    conf, gap = m["Confident (low SC, low SE)"], m["Gap-like (high SC, high SE)"]
    silent = int(y[conf].sum())
    P(f"    Silent errors (wrong inside Confident): {silent} of {y.sum()} "
      f"total errors ({100*silent/max(y.sum(),1):.1f}%), "
      f"= {100*silent/n:.1f}% of all questions")
    # What the thesis did (used labels) vs the label-free version
    old_ans = ~(gap & (y == 1))
    old_acc = 1 - y[old_ans].mean()
    new_ans = ~gap
    new_acc = 1 - y[new_ans].mean()
    P(f"    OLD policy (abstain on Gap-like AND wrong -> uses labels, NOT valid): "
      f"acc {old_acc:.3f}, coverage {old_ans.mean():.3f}")
    P(f"    NEW label-free policy (abstain on Gap-like): "
      f"acc {new_acc:.3f}, coverage {new_ans.mean():.3f}")


# ----------------- 3. policy vs thresholds at matched coverage (key test)
def policy_eval(d):
    y, sc, se, lp, nli = d["wrong"], d["sc"], d["se"], d["lp"], d["nli"]
    n = len(y)
    res = {"all": [], "covA": [], "accA": [], "covB": [], "accB": []}
    base = {}
    for _ in range(REPS):
        idx = rng.integers(0, n, n)
        perm = rng.permutation(n)
        cal, ev = idx[perm[: n // 2]], idx[perm[n // 2:]]
        t_sc, t_se = np.median(sc[cal]), np.median(se[cal])
        hh = (sc[ev] > t_sc) & (se[ev] > t_se)
        ll = (sc[ev] <= t_sc) & (se[ev] <= t_se)
        ye = y[ev]
        kA, kB = int((~hh).sum()), int(ll.sum())
        if kA == 0 or kB == 0:
            continue
        scores = {"SC only": sc[ev], "SE only": se[ev], "LogProb only": lp[ev],
                  "Combined SC+SE+LP": comb([sc, se, lp], cal)[ev]}
        if nli is not None:
            scores["NLI only"] = nli[ev]
            scores["Combined +NLI"] = comb([sc, se, lp, nli], cal)[ev]
        res["all"].append(1 - ye.mean())
        res["covA"].append(kA / len(ye))
        res["accA"].append(1 - ye[~hh].mean())
        res["covB"].append(kB / len(ye))
        res["accB"].append(1 - ye[ll].mean())
        for key, s in scores.items():
            order = np.lexsort((rng.random(len(s)), s))   # random tie-break
            base.setdefault(("A", key), []).append(1 - ye[order[:kA]].mean())
            base.setdefault(("B", key), []).append(1 - ye[order[:kB]].mean())

    ci = lambda v: np.percentile(v, [2.5, 97.5])
    P(f"  [Policy vs thresholds, {len(res['all'])} bootstrap + 50/50 "
      f"calibration/evaluation splits; thresholds fitted on calibration half]")
    P(f"    Answer-all accuracy: {np.mean(res['all']):.3f}")
    labels = {"A": "Fingerprint policy A: abstain only on Gap-like",
              "B": "Fingerprint policy B: auto-accept only Confident"}
    for pol in ("A", "B"):
        cov, acc = np.array(res["cov" + pol]), np.array(res["acc" + pol])
        lo, hi = ci(acc)
        P(f"    {labels[pol]}")
        P(f"      coverage {cov.mean():.3f} | accuracy {acc.mean():.3f} "
          f"[{lo:.3f}, {hi:.3f}]")
        for (p, key), v in base.items():
            if p != pol:
                continue
            v = np.array(v)
            dlo, dhi = ci(acc - v)
            P(f"      vs {key:18s} same coverage: acc {v.mean():.3f} | "
              f"fingerprint - baseline {np.mean(acc - v):+.3f} "
              f"[{dlo:+.3f}, {dhi:+.3f}]")
    P("    (interval entirely above 0 = fingerprint policy clearly better "
      "than that threshold)")


# --------------------- 4. does SC x SE interaction add information?
def interaction_test(d):
    y = d["wrong"]
    z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
    a, b = z(d["sc"]), z(d["se"])

    def ll(X):
        m = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
        return -log_loss(y, m.predict_proba(X)[:, 1], normalize=False)

    stat = 2 * (ll(np.c_[a, b, a * b]) - ll(np.c_[a, b]))
    p = chi2.sf(max(stat, 0.0), 1)
    P(f"  [Interaction SC x SE: likelihood-ratio stat {stat:.3f}, p = {p:.3f}"
      f"  (p>0.05: the 2-D pattern adds nothing beyond SC and SE separately)]")


# ---------- 5. are silent errors visible to the other signals?
def within_confident(d):
    y, sc, se = d["wrong"], d["sc"], d["se"]
    m = (sc <= np.median(sc)) & (se <= np.median(se))
    ym = y[m]
    P(f"  [Inside Confident region: n={m.sum()}, errors={ym.sum()}]")
    for name in ("lp", "nli"):
        if d[name] is None:
            continue
        if 3 <= ym.sum() < len(ym):
            P(f"    {name.upper():4s} AUROC separating right/wrong inside region: "
              f"{roc_auc_score(ym, d[name][m]):.3f}  (few errors -> very noisy)")
        else:
            P(f"    {name.upper():4s} too few errors to evaluate")


# ------------------------------------------------------------- figures
def risk_coverage(y, score, reps=20):
    n = len(y)
    acc = np.zeros(n)
    for _ in range(reps):
        order = np.lexsort((rng.random(n), score))
        acc += np.cumsum(1 - y[order]) / np.arange(1, n + 1)
    return np.arange(1, n + 1) / n, acc / reps


def fig_risk_coverage(d, title):
    y, n = d["wrong"], len(d["wrong"])
    ref = np.arange(n)
    curves = {"SC": d["sc"], "SE": d["se"], "LogProb": d["lp"],
              "Combined (SC+SE+LP)": comb([d["sc"], d["se"], d["lp"]], ref)}
    if d["nli"] is not None:
        curves["SelfCheckGPT-NLI"] = d["nli"]
    plt.figure(figsize=(4.2, 3.2))
    for k, s in curves.items():
        x, a = risk_coverage(y, s)
        keep = x >= 0.3
        plt.plot(x[keep], a[keep], label=k, lw=1.2)
    t_sc, t_se = np.median(d["sc"]), np.median(d["se"])
    hh = (d["sc"] > t_sc) & (d["se"] > t_se)
    ll = (d["sc"] <= t_sc) & (d["se"] <= t_se)
    plt.scatter([(~hh).mean()], [1 - y[~hh].mean()], marker="*", s=90,
                c="k", zorder=5, label="Fingerprint A")
    plt.scatter([ll.mean()], [1 - y[ll].mean()], marker="D", s=40,
                c="k", zorder=5, label="Fingerprint B")
    plt.axhline(1 - y.mean(), ls=":", c="gray", lw=0.8)
    plt.xlabel("Coverage")
    plt.ylabel("Accuracy on answered")
    plt.title(title, fontsize=9)
    plt.legend(fontsize=6, loc="lower left")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(OUT, f"fig_risk_coverage.{ext}"), dpi=300)
    plt.close()


def fig_regions(data):
    names = [n for n in TRIVIA if n in data]
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.2))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        d = data[name]
        y, sc, se = d["wrong"], d["sc"], d["se"]
        hsc, hse = sc > np.median(sc), se > np.median(se)
        cells = [[~hsc & ~hse, ~hsc & hse], [hsc & ~hse, hsc & hse]]
        grid = np.full((2, 2), np.nan)
        for i in range(2):
            for j in range(2):
                m = cells[i][j]
                if m.sum():
                    grid[i, j] = y[m].mean()
        ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn_r")
        for i in range(2):
            for j in range(2):
                m = cells[i][j]
                label = ("n/a" if m.sum() == 0
                         else f"{100 * grid[i, j]:.0f}%")
                ax.text(j, i, f"{label}\n(n={int(m.sum())})",
                        ha="center", va="center", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["low SE", "high SE"], fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["low SC", "high SC"], fontsize=8)
        ax.set_title(name.replace(" | ", "\n"), fontsize=8)
    fig.suptitle("Error rate per fingerprint region", fontsize=9)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(OUT, f"fig_fingerprint_regions.{ext}"), dpi=300)
    plt.close()


# ----------------------------------------------------------------- main
data = {}
for name, path in SOURCES.items():
    if not os.path.exists(path):
        P("MISSING FILE:", path)
        continue
    data[name] = load(path, name)

for name, d in data.items():
    P("=" * 72)
    P(name, "| n =", len(d["wrong"]),
      "| accuracy =", f"{1 - d['wrong'].mean():.3f}")
    if d["nli"] is None:
        P("  NOTE: no SelfCheckGPT-NLI scores in this file")
    table_auc(d)
    regions_report(d)
    interaction_test(d)
    within_confident(d)
    policy_eval(d)

P("=" * 72)
P("SILENT ERRORS ACROSS MODEL SIZE (wrong answers in Confident region)")
for name in TRIVIA:
    if name not in data:
        continue
    d = data[name]
    m = (d["sc"] <= np.median(d["sc"])) & (d["se"] <= np.median(d["se"]))
    k, n = int(d["wrong"][m].sum()), len(d["wrong"])
    lo, hi = wilson(k, n)
    P(f"  {name:24s} {k:3d}/{n} = {100*k/n:4.1f}% of questions "
      f"[{100*lo:.1f}, {100*hi:.1f}]")

first = "Mistral-7B | TriviaQA"
if first in data:
    fig_risk_coverage(data[first], first)
fig_regions(data)
P("\nDone. Files are in the folder:", OUT)
