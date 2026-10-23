"""
analysis3.py - is the "fingerprint policy does not beat thresholds" result
robust to the choice of threshold?

Instead of the median, the SC/SE cut-offs are set at the 30th, 40th, 50th, 60th
and 70th percentile (fitted on the calibration half). For each cut-off the
Gap-like abstention policy is compared with single-score / combined thresholds
at the SAME coverage.

Run from the project folder:
    .\\venv\\Scripts\\python.exe analysis3.py
    .\\venv\\Scripts\\python.exe analysis3.py --judge-labels     (judge labels)
Output: analysis_out/summary_threshold.txt (or analysis_out_judge/...)
"""
import json
import os
import pickle
import sys
import numpy as np

JUDGE = "--judge-labels" in sys.argv
OUT = "analysis_out_judge" if JUDGE else "analysis_out"
os.makedirs(OUT, exist_ok=True)
LABELS = None
if JUDGE:
    with open(os.path.join("analysis_out", "judge_labels.json"), encoding="utf-8") as f:
        LABELS = json.load(f)
log = open(os.path.join(OUT, "summary_threshold.txt"), "w", encoding="utf-8")
rng = np.random.default_rng(42)
REPS = 300
QS = [30, 40, 50, 60, 70]


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


def lab(name, r):
    if LABELS is None:
        return r["is_correct"]
    default = "CORRECT" if r["is_correct"] else "INCORRECT"
    return LABELS.get(f"{name}#{r['idx']}", default) == "CORRECT"


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


def comb(arrs, ref):
    out = []
    for a in arrs:
        lo, hi = a[ref].min(), a[ref].max()
        out.append(np.clip((a - lo) / (hi - lo + 1e-12), 0, 1))
    return np.mean(out, axis=0)


total = above = below = 0
for name, path in SOURCES.items():
    if not os.path.exists(path):
        P("MISSING FILE:", path)
        continue
    d = load(path, name)
    y, sc, se, lp, nli = d["wrong"], d["sc"], d["se"], d["lp"], d["nli"]
    n = len(y)
    store = {q: {"cov": [], "acc": [], "base": {}} for q in QS}
    for _ in range(REPS):
        idx = rng.integers(0, n, n)
        perm = rng.permutation(n)
        cal, ev = idx[perm[: n // 2]], idx[perm[n // 2:]]
        ye = y[ev]
        scores = {"SC only": sc[ev], "SE only": se[ev],
                  "Comb SC+SE+LP": comb([sc, se, lp], cal)[ev]}
        if nli is not None:
            scores["NLI only"] = nli[ev]
        for q in QS:
            t_sc, t_se = np.percentile(sc[cal], q), np.percentile(se[cal], q)
            hh = (sc[ev] > t_sc) & (se[ev] > t_se)
            k = int((~hh).sum())
            if k == 0:
                continue
            acc = 1 - ye[~hh].mean()
            store[q]["cov"].append(k / len(ye))
            store[q]["acc"].append(acc)
            for key, s in scores.items():
                order = np.lexsort((rng.random(len(s)), s))
                store[q]["base"].setdefault(key, []).append(
                    1 - ye[order[:k]].mean())
    P("=" * 78)
    P(f"{name} | n={n} | accuracy={1 - y.mean():.3f}")
    P("  Gap-like abstention at different percentile cut-offs; "
      "difference = policy - baseline at same coverage")
    for q in QS:
        cov, acc = np.array(store[q]["cov"]), np.array(store[q]["acc"])
        parts = []
        for key, v in store[q]["base"].items():
            diff = acc - np.array(v)
            lo, hi = np.percentile(diff, [2.5, 97.5])
            total += 1
            flag = ""
            if lo > 0:
                above += 1
                flag = " (+)"
            elif hi < 0:
                below += 1
                flag = " (-)"
            parts.append(f"{key} {diff.mean():+.3f}[{lo:+.3f},{hi:+.3f}]{flag}")
        P(f"  p{q}: coverage {cov.mean():.2f} acc {acc.mean():.3f} | "
          + " | ".join(parts))

P("=" * 78)
P(f"Across all sources, cut-offs and baselines: {total} comparisons")
P(f"  interval entirely above 0 (policy better): {above}")
P(f"  interval entirely below 0 (policy worse):  {below}")
P(f"  interval includes 0 (no clear difference): {total - above - below}")
P("\nDone. File:", os.path.join(OUT, "summary_threshold.txt"))
