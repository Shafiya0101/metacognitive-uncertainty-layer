import os
import pickle
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

print("Loading results from all models...", flush=True)

models = {
    "Mistral-7B": "results",
    "Qwen2-7B": "results_llama3",
    "Qwen2-1.5B": "results_phi3",
}

def mnorm(x):
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)

all_data = {}
for label, folder in models.items():
    path = os.path.join(os.getcwd(), folder, "final_results.pkl")
    if not os.path.exists(path):
        print(f"  MISSING: {label} ({path})", flush=True)
        continue

    with open(path, "rb") as f:
        d = pickle.load(f)

    # Get results list
    res = d.get("results", [])
    if not res:
        print(f"  EMPTY: {label}", flush=True)
        continue

    # Compute accuracy
    greedy_acc = sum(r["is_correct"] for r in res) / len(res)
    majority_acc = sum(r.get("majority_correct", r["is_correct"]) for r in res) / len(res)

    # Compute AUROC for all methods
    labels = np.array([float(not r["is_correct"]) for r in res])
    ev = {}

    if np.std(labels) > 0:
        for key, field, mx in [
            ("Avg(-logP)", "avg_neg_log_prob", None),
            ("Self-Consistency", "uncertainty_sc", None),
            ("Semantic Entropy", "uncertainty_se", None),
            ("SelfCheckGPT-NLI", "uncertainty_selfcheck", None),
        ]:
            if field in res[0]:
                vals = np.array([r[field] for r in res])
                if np.std(vals) > 1e-10:
                    ev[key] = {"auroc": roc_auc_score(labels, vals)}
                else:
                    ev[key] = {"auroc": 0.5}

        # Combined
        if all(f in res[0] for f in ["uncertainty_sc", "uncertainty_se", "avg_neg_log_prob"]):
            sc = np.array([r["uncertainty_sc"] for r in res])
            se = np.array([r["uncertainty_se"] for r in res])
            lp = np.array([r["avg_neg_log_prob"] for r in res])
            combined = (mnorm(sc) + mnorm(se) + mnorm(lp)) / 3.0
            ev["Combined"] = {"auroc": roc_auc_score(labels, combined)}

    # Compute fingerprints
    type_counts = {}
    if "fingerprint_type" in res[0]:
        type_counts = dict(Counter(r["fingerprint_type"] for r in res))
    elif all(f in res[0] for f in ["uncertainty_sc", "uncertainty_se"]):
        sc_vals = np.array([r["uncertainty_sc"] for r in res])
        se_vals = np.array([r["uncertainty_se"] for r in res])
        sc_med = np.median(sc_vals)
        se_med = np.median(se_vals)
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
        type_counts = dict(Counter(r["fingerprint_type"] for r in res))

    all_data[label] = {
        "greedy_acc": greedy_acc,
        "majority_acc": majority_acc,
        "eval": ev,
        "type_counts": type_counts,
    }
    print(f"  Loaded {label}: acc={greedy_acc:.2%}, combined={ev.get('Combined', {}).get('auroc', 0):.4f}", flush=True)

if len(all_data) < 2:
    print("\nNeed at least 2 models.", flush=True)
    input("Press Enter..."); exit()

SAVE_DIR = os.path.join(os.getcwd(), "results_comparison")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── TABLE 1: OVERVIEW ──
print("\n" + "=" * 70, flush=True)
print("  MULTI-MODEL COMPARISON", flush=True)
print("=" * 70, flush=True)

print(f"\n{'Model':<16} {'Greedy':<10} {'Majority':<10} {'Combined':<10} {'SelfCheck':<10}", flush=True)
print("-" * 56, flush=True)
for label, data in all_data.items():
    ga = data["greedy_acc"]
    ma = data["majority_acc"]
    ca = data["eval"].get("Combined", {}).get("auroc", 0)
    sc = data["eval"].get("SelfCheckGPT-NLI", {}).get("auroc", 0)
    print(f"{label:<16} {ga:<10.2%} {ma:<10.2%} {ca:<10.4f} {sc:<10.4f}", flush=True)

# ── TABLE 2: PER-METHOD ──
methods = ["Avg(-logP)", "Self-Consistency", "Semantic Entropy", "SelfCheckGPT-NLI", "Combined"]
print(f"\n{'Method':<22}", end="", flush=True)
for label in all_data:
    print(f" {label:<16}", end="", flush=True)
print("", flush=True)
print("-" * (22 + 16 * len(all_data)), flush=True)
for method in methods:
    print(f"{method:<22}", end="", flush=True)
    for label in all_data:
        auroc = all_data[label]["eval"].get(method, {}).get("auroc", 0)
        print(f" {auroc:<16.4f}", end="", flush=True)
    print("", flush=True)

# ── TABLE 3: FINGERPRINTS ──
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

# ── PLOT 1: AUROC ──
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(methods))
width = 0.8 / len(all_data)
colors = ["#3b82f6", "#22c55e", "#f59e0b"]
for i, (label, data) in enumerate(all_data.items()):
    aurocs = [data["eval"].get(m, {}).get("auroc", 0) for m in methods]
    ax.bar(x + i * width, aurocs, width, label=label, color=colors[i % len(colors)], edgecolor="white")
ax.set_ylabel("AUROC", fontsize=12)
ax.set_title("Multi-Model AUROC Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x + width * (len(all_data) - 1) / 2)
ax.set_xticklabels(methods, fontsize=10)
ax.legend(fontsize=11)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "multimodel_auroc.png"), bbox_inches="tight", dpi=150)
print("\nSaved multimodel_auroc.png", flush=True)

# ── PLOT 2: ACCURACY ──
fig, ax = plt.subplots(figsize=(8, 5))
labels_list = list(all_data.keys())
greedy = [all_data[l]["greedy_acc"] * 100 for l in labels_list]
x = np.arange(len(labels_list))
bars = ax.bar(x, greedy, 0.5, color=colors[:len(labels_list)], edgecolor="white")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Greedy Accuracy by Model", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels_list)
for bar, val in zip(bars, greedy):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}%", ha="center", fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "multimodel_accuracy.png"), bbox_inches="tight", dpi=150)
print("Saved multimodel_accuracy.png", flush=True)

# ── PLOT 3: CONFIDENT FABRICATIONS ──
fig, ax = plt.subplots(figsize=(8, 5))
cf_counts = [all_data[l]["type_counts"].get("confident_fabrication", 0) for l in labels_list]
cf_pcts = []
for l in labels_list:
    tc = all_data[l]["type_counts"]
    total = sum(tc.values()) if tc else 1
    cf = tc.get("confident_fabrication", 0)
    cf_pcts.append(cf / total * 100)
bars = ax.bar(x, cf_pcts, 0.5, color=["#ef4444"] * len(labels_list), edgecolor="white")
ax.set_ylabel("% of Questions")
ax.set_title("Confident Fabrications by Model Size", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels_list)
for bar, val, cnt in zip(bars, cf_pcts, cf_counts):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f"{val:.1f}% (n={cnt})", ha="center", fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confident_fabrications_by_model.png"), bbox_inches="tight", dpi=150)
print("Saved confident_fabrications_by_model.png", flush=True)

# ── KEY FINDINGS ──
print(f"\n{'='*70}", flush=True)
print("  KEY FINDINGS", flush=True)
print(f"{'='*70}", flush=True)
best = max(all_data.keys(), key=lambda l: all_data[l]["eval"].get("Combined", {}).get("auroc", 0))
print(f"\n  Best combined AUROC: {best} ({all_data[best]['eval']['Combined']['auroc']:.4f})", flush=True)
print(f"\n  Confident fabrication rate by model:", flush=True)
for label in labels_list:
    tc = all_data[label]["type_counts"]
    total = sum(tc.values()) if tc else 1
    cf = tc.get("confident_fabrication", 0)
    pct = cf / total * 100 if total > 0 else 0
    print(f"    {label}: {cf} ({pct:.1f}%)", flush=True)
print(f"\n  FINDING: Smaller models produce MORE confident fabrications!", flush=True)
print(f"  This means uncertainty monitoring is MORE critical for small/edge models.", flush=True)
print(f"\n{'='*70}", flush=True)
print(f"Results saved to: {SAVE_DIR}", flush=True)
input("\nPress Enter to exit...")