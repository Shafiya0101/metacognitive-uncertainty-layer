"""
analysis2.py - silent errors with ABSOLUTE (not median) definitions.

Run from the project folder:
    .\\venv\\Scripts\\python.exe analysis2.py

Why: with median thresholds, "Confident" can mean SC <= 0.7 (see Qwen2-1.5B),
which is not confident at all. Here "confident" is defined in absolute terms:
  strict          : all 10 samples give the same answer (SC = 0, SE = 0)
  high agreement  : at least 9 of 10 samples agree      (SC <= 0.1)
Output: analysis_out/summary_absolute.txt
"""
import os
import pickle
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import json
import sys
JUDGE = "--judge-labels" in sys.argv     # use Claude-judge labels instead of containment matching
OUT = "analysis_out_judge" if JUDGE else "analysis_out"
os.makedirs(OUT, exist_ok=True)
LABELS = None
if JUDGE:
    with open(os.path.join("analysis_out", "judge_labels.json"), encoding="utf-8") as _f:
        LABELS = json.load(_f)
log = open(os.path.join(OUT, "summary_absolute.txt"), "w", encoding="utf-8")


def lab(name, r):
    if LABELS is None:
        return r["is_correct"]
    default = "CORRECT" if r["is_correct"] else "INCORRECT"
    return LABELS.get(f"{name}#{r['idx']}", default) == "CORRECT"


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


def load(path, name):
    with open(path, "rb") as f:
        d = pickle.load(f)
    res = d["results"] if isinstance(d, dict) else d

    def col(k):
        if k not in res[0]:
            return None
        return np.array([r[k] for r in res], dtype=float)

    return {"wrong": np.array([0 if lab(name, r) else 1 for r in res]),
            "sc": col("uncertainty_sc"), "se": col("uncertainty_se"),
            "lp": col("avg_neg_log_prob"), "nli": col("uncertainty_selfcheck")}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


summary = []
for name, path in SOURCES.items():
    if not os.path.exists(path):
        P("MISSING FILE:", path)
        continue
    d = load(path, name)
    y, sc, se = d["wrong"], d["sc"], d["se"]
    n = len(y)
    rho = spearmanr(sc, se)[0]
    P("=" * 72)
    P(f"{name} | n={n} | accuracy={1 - y.mean():.3f} | "
      f"total errors={int(y.sum())}")
    P(f"  Spearman correlation between SC and SE: {rho:.3f} "
      f"(close to 1 = the two signals are largely redundant)")
    defs = [
        ("strict: all 10 samples agree (SC=0, SE=0)",
         (sc <= 1e-9) & (np.abs(se) <= 1e-6)),
        ("high agreement: >=9/10 samples agree (SC<=0.1)", sc <= 0.1 + 1e-9),
    ]
    for label, m in defs:
        nm, k = int(m.sum()), int(y[m].sum())
        r_lo, r_hi = wilson(k, nm)
        s_lo, s_hi = wilson(k, n)
        P(f"  [{label}]")
        P(f"    questions in region: {nm} ({100 * nm / n:.1f}%)")
        if nm:
            P(f"    wrong inside region: {k}  -> error rate {k / nm:.3f} "
              f"[{r_lo:.3f}, {r_hi:.3f}]")
        P(f"    silent errors = {100 * k / n:.1f}% of all questions "
          f"[{100 * s_lo:.1f}, {100 * s_hi:.1f}] ; "
          f"{100 * k / max(y.sum(), 1):.1f}% of all errors")
        ym = y[m]
        for sig in ("lp", "nli"):
            if d[sig] is None:
                continue
            if ym.sum() >= 5 and ym.sum() < len(ym):
                P(f"    {sig.upper():3s} AUROC for right/wrong inside region: "
                  f"{roc_auc_score(ym, d[sig][m]):.3f} (n_errors={int(ym.sum())})")
            else:
                P(f"    {sig.upper():3s} AUROC: too few errors "
                  f"({int(ym.sum())}) to evaluate")
        if label.startswith("strict"):
            summary.append((name, nm, k, n))

P("=" * 72)
P("STRICT SILENT ERRORS (all 10 samples agree, answer wrong)")
for name, nm, k, n in summary:
    lo, hi = wilson(k, n)
    P(f"  {name:24s} {k:3d}/{n} = {100 * k / n:4.1f}% of questions "
      f"[{100 * lo:.1f}, {100 * hi:.1f}]   (region size {nm})")
P("\nDone. File:", os.path.join(OUT, "summary_absolute.txt"))
