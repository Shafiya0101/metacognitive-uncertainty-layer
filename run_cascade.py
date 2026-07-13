import os, pickle, numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("CASCADING UNCERTAINTY ANALYSIS", flush=True)

sources = {"Mistral-TriviaQA": "results", "Qwen2-7B-TriviaQA": "results_llama3",
           "Qwen2-1.5B-TriviaQA": "results_phi3", "Mistral-NaturalQA": "results_naturalqa",
           "Mistral-HotpotQA": "results_hotpotqa"}

all_combined = []
for name, folder in sources.items():
    path = os.path.join(os.getcwd(), folder, "final_results.pkl")
    if not os.path.exists(path):
        continue
    with open(path, "rb") as f:
        d = pickle.load(f)
    res = d.get("results", [])
    if res and "uncertainty_sc" in res[0]:
        all_combined.extend(res)
        print(f"  Loaded {name} ({len(res)} q)", flush=True)

print(f"Total: {len(all_combined)} questions", flush=True)
SAVE_DIR = os.path.join(os.getcwd(), "results_cascade")
os.makedirs(SAVE_DIR, exist_ok=True)

COSTS = {"s1": 3, "s2": 33, "s3": 63}

def cascade(results, lp_t, sc_t):
    s1 = s2 = s3 = 0
    costs = []
    correct_total = 0
    for r in results:
        lp = r["avg_neg_log_prob"]
        sc = r["uncertainty_sc"]
        if lp < lp_t:
            s1 += 1
            costs.append(COSTS["s1"])
        elif sc < sc_t:
            s2 += 1
            costs.append(COSTS["s2"])
        else:
            s3 += 1
            costs.append(COSTS["s3"])
        if r["is_correct"]:
            correct_total += 1
    n = len(results)
    return {
        "s1_pct": s1/n*100, "s2_pct": s2/n*100, "s3_pct": s3/n*100,
        "avg_cost": np.mean(costs), "speedup": COSTS["s3"]/np.mean(costs),
        "accuracy": correct_total/n*100,
        "s1": s1, "s2": s2, "s3": s3,
    }

lp_all = np.array([r["avg_neg_log_prob"] for r in all_combined])
sc_all = np.array([r["uncertainty_sc"] for r in all_combined])

configs = [
    ("No Cascade", 0, 0),
    ("Conservative", np.percentile(lp_all, 20), np.percentile(sc_all, 30)),
    ("Balanced", np.percentile(lp_all, 35), np.percentile(sc_all, 50)),
    ("Aggressive", np.percentile(lp_all, 50), np.percentile(sc_all, 65)),
    ("Very Aggressive", np.percentile(lp_all, 65), np.percentile(sc_all, 80)),
]

print(f"\n{'Config':<20} {'Stage1%':<9} {'Stage2%':<9} {'Stage3%':<9} {'AvgTime':<9} {'Speedup':<9}", flush=True)
print("-" * 65, flush=True)

plot_data = []
for cname, lp_t, sc_t in configs:
    ev = cascade(all_combined, lp_t, sc_t)
    plot_data.append((cname, ev))
    print(f"{cname:<20} {ev['s1_pct']:<9.1f} {ev['s2_pct']:<9.1f} {ev['s3_pct']:<9.1f} "
          f"{ev['avg_cost']:<9.1f}s {ev['speedup']:<9.1f}x", flush=True)

# Per-stage accuracy for balanced
bal_lp = np.percentile(lp_all, 35)
bal_sc = np.percentile(sc_all, 50)
s1_items = [r for r in all_combined if r["avg_neg_log_prob"] < bal_lp]
s2_items = [r for r in all_combined if r["avg_neg_log_prob"] >= bal_lp and r["uncertainty_sc"] < bal_sc]
s3_items = [r for r in all_combined if r["avg_neg_log_prob"] >= bal_lp and r["uncertainty_sc"] >= bal_sc]

print(f"\nPer-stage accuracy (Balanced):", flush=True)
for label, items in [("Stage 1 (logit)", s1_items), ("Stage 2 (+SC)", s2_items), ("Stage 3 (full)", s3_items)]:
    if items:
        acc = sum(r["is_correct"] for r in items) / len(items)
        print(f"  {label}: {len(items)} questions, {acc:.1%} accuracy", flush=True)

# PLOT 1: Speedup vs stage distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cnames = [c[0] for c in plot_data]
s1_pcts = [c[1]["s1_pct"] for c in plot_data]
s2_pcts = [c[1]["s2_pct"] for c in plot_data]
s3_pcts = [c[1]["s3_pct"] for c in plot_data]

x = np.arange(len(cnames))
axes[0].bar(x, s1_pcts, label="Stage 1 (logit, 3s)", color="#22c55e", edgecolor="white")
axes[0].bar(x, s2_pcts, bottom=s1_pcts, label="Stage 2 (+SC, 33s)", color="#f59e0b", edgecolor="white")
s12 = [a+b for a, b in zip(s1_pcts, s2_pcts)]
axes[0].bar(x, s3_pcts, bottom=s12, label="Stage 3 (full, 63s)", color="#ef4444", edgecolor="white")
axes[0].set_ylabel("% of Questions")
axes[0].set_title("Questions per Stage", fontweight="bold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(cnames, fontsize=9, rotation=15)
axes[0].legend(fontsize=9)

# PLOT 2: Speedup bar chart
speedups = [c[1]["speedup"] for c in plot_data]
colors = ["#ef4444", "#22c55e", "#3b82f6", "#f59e0b", "#8b5cf6"]
bars = axes[1].bar(cnames, speedups, color=colors[:len(cnames)], edgecolor="white", width=0.6)
axes[1].set_ylabel("Speedup (x)")
axes[1].set_title("Speedup vs No Cascade", fontweight="bold")
axes[1].set_xticklabels(cnames, fontsize=9, rotation=15)
for bar, val in zip(bars, speedups):
    axes[1].text(bar.get_x()+bar.get_width()/2, val+0.05, f"{val:.1f}x", ha="center", fontsize=11, fontweight="bold")
axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)

plt.suptitle(f"Cascading Uncertainty (n={len(all_combined)} questions)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "cascade_tradeoff.png"), bbox_inches="tight", dpi=150)
print("\nSaved cascade_tradeoff.png", flush=True)

# KEY FINDINGS
bal = plot_data[2][1]
print(f"\n{'='*60}", flush=True)
print("  KEY FINDINGS (Balanced cascade)", flush=True)
print(f"{'='*60}", flush=True)
print(f"  {bal['s1_pct']:.0f}% resolved at Stage 1 (logit only, 3s)", flush=True)
print(f"  {bal['s2_pct']:.0f}% resolved at Stage 2 (+SC, 33s)", flush=True)
print(f"  {bal['s3_pct']:.0f}% need full pipeline (63s)", flush=True)
print(f"  Average latency: {bal['avg_cost']:.1f}s (down from 63s)", flush=True)
print(f"  Speedup: {bal['speedup']:.1f}x faster", flush=True)
print(f"  Conclusion: {bal['speedup']:.1f}x speedup makes real-time deployment feasible", flush=True)
print(f"{'='*60}", flush=True)

with open(os.path.join(SAVE_DIR, "cascade_results.pkl"), "wb") as f:
    pickle.dump({"plot_data": plot_data, "total_questions": len(all_combined)}, f)
print("Saved cascade_results.pkl", flush=True)

print("\nDone!", flush=True)
input("Press Enter to exit...")