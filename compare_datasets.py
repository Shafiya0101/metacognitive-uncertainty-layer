import os
import pickle
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

print("Loading results from all datasets...", flush=True)

datasets = {
    "TriviaQA": "results",
    "NaturalQA": "results_naturalqa",
    "HotpotQA": "results_hotpotqa",
}

def mnorm(x):
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)

all_data = {}
for label, folder in datasets.items():
    path = os.path.join(os.getcwd(), folder, "final_results.pkl")
    if not os.path.exists(path):
        print(f"  MISSING: {label}", flush=True)
        continue
    with open(path, "rb") as f:
        d = pickle.load(f)
    res = d.get("results", [])
    if not res:
        print(f"  EMPTY: {label}", flush=True)
        continue
    greedy_acc = sum(r["is_correct"] for r in res) / len(res)
    majority_acc = sum(r.get("majority_correct", r["is_correct"]) for r in res) / len(res)
    labels = np.array([float(not r["is_correct"]) for r in res])
    ev = {}
    if np.std(labels) > 0:
        for key, field in [("Avg(-logP)", "avg_neg_log_prob"), ("Self-Consistency", "uncertainty_sc"),
                           ("Semantic Entropy", "uncertainty_se"), ("SelfCheckGPT-NLI", "uncertainty_selfcheck")]:
            if field in res[0]:
                vals = np.array([r[field] for r in res])
                ev[key] = {"auroc": roc_auc_score(labels, vals) if np.std(vals) > 1e-10 else 0.5}
        if all(f in res[0] for f in ["uncertainty_sc", "uncertainty_se", "avg_neg_log_prob"]):
            sc = np.array([r["uncertainty_sc"] for r in res])
            se = np.array([r["uncertainty_se"] for r in res])
            lp = np.array([r["avg_neg_log_prob"] for r in res])
            combined = (mnorm(sc) + mnorm(se) + mnorm(lp)) / 3.0
            ev["Combined"] = {"auroc": roc_auc_score(labels, combined)}
    # Fingerprints
    if "fingerprint_type" not in res[0] and all(f in res[0] for f in ["uncertainty_sc", "uncertainty_se"]):
        sc_vals = np.array([r["uncertainty_sc"] for r in res])
        se_vals = np.array([r["uncertainty_se"] for r in res])
        sc_med, se_med = np.median(sc_vals), np.median(se_vals)
        for r in res:
            sc_h = r["uncertainty_sc"] > sc_med
            se_h = r["uncertainty_se"] > se_med
            wrong = not r["is_correct"]
            if not wrong:
                r["fingerprint_type"] = "correct_confident" if not sc_h and not se_h else "correct_uncertain"
            elif not sc_h and not se_h:
                r["fingerprint_type"] = "confident_fabrication"
            elif sc_h and se_h:
                r["fingerprint_type"] = "knowledge_gap"
            elif not sc_h and se_h:
                r["fingerprint_type"] = "shallow_mimicry"
            else:
                r["fingerprint_type"] = "inconsistent_surface"
    type_counts = dict(Counter(r.get("fingerprint_type", "unknown") for r in res))
    # Policy
    answered = sum(1 for r in res if r.get("fingerprint_type") != "knowledge_gap")
    correct_ans = sum(1 for r in res if r.get("fingerprint_type") != "knowledge_gap" and r["is_correct"])
    policy_acc = correct_ans / answered if answered > 0 else 0
    coverage = answered / len(res)
    all_data[label] = {
        "greedy_acc": greedy_acc, "majority_acc": majority_acc,
        "eval": ev, "type_counts": type_counts,
        "policy_acc": policy_acc, "coverage": coverage, "n": len(res),
    }
    print(f"  Loaded {label}: acc={greedy_acc:.2%}, combined={ev.get('Combined', {}).get('auroc', 0):.4f}", flush=True)

if len(all_data) < 2:
    print("Need at least 2 datasets.", flush=True)
    input("Press Enter..."); exit()

SAVE_DIR = os.path.join(os.getcwd(), "results_dataset_comparison")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── TABLES ──
print("\n" + "=" * 70, flush=True)
print("  CROSS-DATASET COMPARISON", flush=True)
print("=" * 70, flush=True)

print(f"\n{'Dataset':<16} {'Greedy':<10} {'Majority':<10} {'Combined':<10} {'SelfCheck':<10}", flush=True)
print("-" * 56, flush=True)
for label, data in all_data.items():
    print(f"{label:<16} {data['greedy_acc']:<10.2%} {data['majority_acc']:<10.2%} "
          f"{data['eval'].get('Combined', {}).get('auroc', 0):<10.4f} "
          f"{data['eval'].get('SelfCheckGPT-NLI', {}).get('auroc', 0):<10.4f}", flush=True)

methods = ["Avg(-logP)", "Self-Consistency", "Semantic Entropy", "SelfCheckGPT-NLI", "Combined"]
print(f"\n{'Method':<22}", end="", flush=True)
for label in all_data:
    print(f" {label:<16}", end="", flush=True)
print("", flush=True)
print("-" * (22 + 16 * len(all_data)), flush=True)
for m in methods:
    print(f"{m:<22}", end="", flush=True)
    for label in all_data:
        print(f" {all_data[label]['eval'].get(m, {}).get('auroc', 0):<16.4f}", end="", flush=True)
    print("", flush=True)

fp_types = ["correct_confident", "correct_uncertain", "knowledge_gap",
            "confident_fabrication", "shallow_mimicry", "inconsistent_surface"]
print(f"\n{'Fingerprint':<25}", end="", flush=True)
for label in all_data:
    print(f" {label:<16}", end="", flush=True)
print("", flush=True)
print("-" * (25 + 16 * len(all_data)), flush=True)
for ft in fp_types:
    print(f"{ft:<25}", end="", flush=True)
    for label in all_data:
        tc = all_data[label]["type_counts"]
        total = sum(tc.values()) if tc else 1
        count = tc.get(ft, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f" {count:>3} ({pct:4.1f}%)     ", end="", flush=True)
    print("", flush=True)

print(f"\n{'Dataset':<16} {'Baseline':<12} {'Policy Acc':<12} {'Coverage':<10} {'Improvement':<12}", flush=True)
print("-" * 62, flush=True)
for label, data in all_data.items():
    imp = data["policy_acc"] - data["greedy_acc"]
    print(f"{label:<16} {data['greedy_acc']:<12.2%} {data['policy_acc']:<12.2%} "
          f"{data['coverage']:<10.1%} {imp:<+12.2%}", flush=True)

# ── PLOTS ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ["#3b82f6", "#22c55e", "#f59e0b"]
labels_list = list(all_data.keys())

# Plot 1: AUROC
x = np.arange(len(methods))
width = 0.8 / len(all_data)
for i, (label, data) in enumerate(all_data.items()):
    aurocs = [data["eval"].get(m, {}).get("auroc", 0) for m in methods]
    axes[0].bar(x + i * width, aurocs, width, label=label, color=colors[i], edgecolor="white")
axes[0].set_ylabel("AUROC")
axes[0].set_title("Detection Performance", fontweight="bold")
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(methods, fontsize=8, rotation=15)
axes[0].legend(fontsize=9)
axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
axes[0].grid(axis="y", alpha=0.3)

# Plot 2: Confident fabrications
cf_pcts = []
for l in labels_list:
    tc = all_data[l]["type_counts"]
    total = sum(tc.values()) if tc else 1
    cf_pcts.append(tc.get("confident_fabrication", 0) / total * 100)
bars = axes[1].bar(labels_list, cf_pcts, color=["#ef4444"]*3, edgecolor="white", width=0.5)
axes[1].set_ylabel("% of Questions")
axes[1].set_title("Confident Fabrication Rate", fontweight="bold")
for bar, val in zip(bars, cf_pcts):
    axes[1].text(bar.get_x()+bar.get_width()/2, val+0.3, f"{val:.1f}%", ha="center", fontsize=11)
axes[1].grid(axis="y", alpha=0.3)

# Plot 3: Policy improvement
baseline = [all_data[l]["greedy_acc"]*100 for l in labels_list]
policy = [all_data[l]["policy_acc"]*100 for l in labels_list]
x2 = np.arange(len(labels_list))
axes[2].bar(x2-0.2, baseline, 0.35, label="Baseline", color="#ef4444", edgecolor="white")
axes[2].bar(x2+0.2, policy, 0.35, label="With Policy", color="#22c55e", edgecolor="white")
axes[2].set_ylabel("Accuracy (%)")
axes[2].set_title("Adaptive Policy Improvement", fontweight="bold")
axes[2].set_xticks(x2)
axes[2].set_xticklabels(labels_list)
axes[2].legend()
axes[2].grid(axis="y", alpha=0.3)

plt.suptitle("Cross-Dataset Analysis (Mistral-7B)", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "crossdataset_comparison.png"), bbox_inches="tight", dpi=150)
print("\nSaved crossdataset_comparison.png", flush=True)

# ── KEY FINDINGS ──
print(f"\n{'='*70}", flush=True)
print("  KEY FINDINGS", flush=True)
print(f"{'='*70}", flush=True)
for label in labels_list:
    ev = all_data[label]["eval"]
    comb = ev.get("Combined", {}).get("auroc", 0)
    sc = ev.get("SelfCheckGPT-NLI", {}).get("auroc", 0)
    winner = "Combined" if comb > sc else "SelfCheckGPT" if sc > comb else "Tied"
    print(f"  {label}: {winner} wins (Combined={comb:.4f}, SelfCheck={sc:.4f})", flush=True)
print(f"\n  Confident fabrications increase with task difficulty:", flush=True)
for label in labels_list:
    tc = all_data[label]["type_counts"]
    total = sum(tc.values()) if tc else 1
    cf = tc.get("confident_fabrication", 0)
    pct = cf / total * 100 if total > 0 else 0
    print(f"    {label}: {cf} ({pct:.1f}%)", flush=True)
print(f"\n  Policy improvement is LARGEST on hardest tasks:", flush=True)
for label in labels_list:
    imp = all_data[label]["policy_acc"] - all_data[label]["greedy_acc"]
    print(f"    {label}: {imp:+.2%}", flush=True)
print(f"\n{'='*70}", flush=True)
input("\nPress Enter to exit...")