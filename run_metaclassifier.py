import os
import pickle
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 70, flush=True)
print("  LEARNED META-CLASSIFIER FOR HALLUCINATION DETECTION", flush=True)
print("=" * 70, flush=True)

def mnorm(x):
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)

# Load all available results
sources = {
    "Mistral-TriviaQA": "results",
    "Qwen2-7B-TriviaQA": "results_llama3",
    "Qwen2-1.5B-TriviaQA": "results_phi3",
    "Mistral-NaturalQA": "results_naturalqa",
    "Mistral-HotpotQA": "results_hotpotqa",
}

all_features = []
all_labels = []
all_sources = []
per_source = {}

for name, folder in sources.items():
    path = os.path.join(os.getcwd(), folder, "final_results.pkl")
    if not os.path.exists(path):
        continue
    with open(path, "rb") as f:
        d = pickle.load(f)
    res = d.get("results", [])
    if not res or "uncertainty_sc" not in res[0]:
        continue

    features = []
    labels = []
    for r in res:
        sc = r.get("uncertainty_sc", 0)
        se = r.get("uncertainty_se", 0)
        lp = r.get("avg_neg_log_prob", 0)
        nli = r.get("uncertainty_selfcheck", 0.5)
        features.append([sc, se, lp, nli])
        labels.append(float(not r["is_correct"]))

    features = np.array(features)
    labels = np.array(labels)

    if np.std(labels) == 0:
        continue

    # Equal weight baseline for this source
    combined_equal = (mnorm(features[:, 0]) + mnorm(features[:, 1]) + mnorm(features[:, 2])) / 3.0
    auroc_equal = roc_auc_score(labels, combined_equal)

    # Per-source logistic regression (5-fold CV)
    clf = LogisticRegression(random_state=42, max_iter=1000)
    lr_scores = cross_val_predict(clf, features, labels, cv=5, method="predict_proba")[:, 1]
    auroc_lr = roc_auc_score(labels, lr_scores)

    # Per-source MLP (5-fold CV)
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), random_state=42, max_iter=1000)
    mlp_scores = cross_val_predict(mlp, features, labels, cv=5, method="predict_proba")[:, 1]
    auroc_mlp = roc_auc_score(labels, mlp_scores)

    # Fit LR on full data to get weights
    clf.fit(features, labels)
    weights = clf.coef_[0]

    per_source[name] = {
        "n": len(labels),
        "hall_rate": labels.mean(),
        "auroc_equal": auroc_equal,
        "auroc_lr": auroc_lr,
        "auroc_mlp": auroc_mlp,
        "weights": weights,
    }

    all_features.append(features)
    all_labels.append(labels)
    all_sources.extend([name] * len(labels))

    print(f"\n  {name} (n={len(labels)}, hall_rate={labels.mean():.1%}):", flush=True)
    print(f"    Equal-weight:     AUROC={auroc_equal:.4f}", flush=True)
    print(f"    Logistic Reg(CV): AUROC={auroc_lr:.4f}", flush=True)
    print(f"    MLP(CV):          AUROC={auroc_mlp:.4f}", flush=True)
    print(f"    Learned weights:  SC={weights[0]:+.3f} SE={weights[1]:+.3f} LP={weights[2]:+.3f} NLI={weights[3]:+.3f}", flush=True)

# ── UNIFIED CLASSIFIER (trained on ALL data) ──
if len(all_features) >= 2:
    print(f"\n{'='*70}", flush=True)
    print("  UNIFIED META-CLASSIFIER (all data combined)", flush=True)
    print(f"{'='*70}", flush=True)

    X_all = np.vstack(all_features)
    y_all = np.concatenate(all_labels)
    print(f"\n  Total samples: {len(y_all)}", flush=True)
    print(f"  Hallucination rate: {y_all.mean():.1%}", flush=True)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # Equal weight
    auroc_equal_all = roc_auc_score(y_all, (mnorm(X_all[:, 0]) + mnorm(X_all[:, 1]) + mnorm(X_all[:, 2])) / 3.0)

    # Logistic Regression
    clf_all = LogisticRegression(random_state=42, max_iter=1000)
    lr_all = cross_val_predict(clf_all, X_scaled, y_all, cv=5, method="predict_proba")[:, 1]
    auroc_lr_all = roc_auc_score(y_all, lr_all)
    auprc_lr_all = average_precision_score(y_all, lr_all)

    # MLP
    mlp_all = MLPClassifier(hidden_layer_sizes=(32, 16), random_state=42, max_iter=1000)
    mlp_all_scores = cross_val_predict(mlp_all, X_scaled, y_all, cv=5, method="predict_proba")[:, 1]
    auroc_mlp_all = roc_auc_score(y_all, mlp_all_scores)
    auprc_mlp_all = average_precision_score(y_all, mlp_all_scores)

    # SelfCheckGPT-NLI alone
    auroc_nli_all = roc_auc_score(y_all, X_all[:, 3])

    # Fit on full data for weights
    clf_all.fit(X_scaled, y_all)
    feat_names = ["Self-Consistency", "Semantic Entropy", "Avg(-logP)", "SelfCheckGPT-NLI"]

    print(f"\n  {'Method':<35} {'AUROC':<10} {'AUPRC':<10}", flush=True)
    print("  " + "-" * 55, flush=True)
    print(f"  {'SelfCheckGPT-NLI (single signal)':<35} {auroc_nli_all:<10.4f}", flush=True)
    print(f"  {'Equal-weight (SC+SE+LP)':<35} {auroc_equal_all:<10.4f}", flush=True)
    print(f"  {'Logistic Regression (CV)':<35} {auroc_lr_all:<10.4f} {auprc_lr_all:<10.4f}", flush=True)
    print(f"  {'Neural Network MLP (CV)':<35} {auroc_mlp_all:<10.4f} {auprc_mlp_all:<10.4f}", flush=True)

    print(f"\n  LEARNED SIGNAL IMPORTANCE (Logistic Regression):", flush=True)
    for name, coef in zip(feat_names, clf_all.coef_[0]):
        bar_len = int(abs(coef) * 8)
        bar = "#" * bar_len
        direction = "+" if coef > 0 else "-"
        print(f"    {name:<25} {coef:+.4f}  {bar} ({direction})", flush=True)

    # ── Per-source performance with unified classifier ──
    print(f"\n  UNIFIED CLASSIFIER APPLIED PER SOURCE:", flush=True)
    print(f"  {'Source':<30} {'Equal':<10} {'Unified LR':<12} {'Improvement':<12}", flush=True)
    print("  " + "-" * 64, flush=True)

    start = 0
    for name in per_source:
        n = per_source[name]["n"]
        X_src = X_scaled[start:start+n]
        y_src = y_all[start:start+n]
        if np.std(y_src) > 0:
            unified_scores = clf_all.predict_proba(X_src)[:, 1]
            auroc_unified = roc_auc_score(y_src, unified_scores)
            eq = per_source[name]["auroc_equal"]
            imp = auroc_unified - eq
            print(f"  {name:<30} {eq:<10.4f} {auroc_unified:<12.4f} {imp:<+12.4f}", flush=True)
        start += n

    # ── FINGERPRINT WITH LEARNED THRESHOLDS ──
    print(f"\n{'='*70}", flush=True)
    print("  LEARNED FINGERPRINTING (replaces median thresholds)", flush=True)
    print(f"{'='*70}", flush=True)

    # Use the LR probability as the threshold
    # Low probability = likely correct, High = likely hallucinated
    # But we can also learn optimal thresholds for fingerprinting
    from sklearn.tree import DecisionTreeClassifier

    # Train a decision tree to learn fingerprint boundaries
    # Features: 4 signals, Target: correct vs hallucinated
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_scores = cross_val_predict(dt, X_scaled, y_all, cv=5, method="predict_proba")[:, 1]
    auroc_dt = roc_auc_score(y_all, dt_scores)
    print(f"\n  Decision Tree (learned thresholds, CV): AUROC={auroc_dt:.4f}", flush=True)

    # Show the tree's learned rules
    dt.fit(X_scaled, y_all)
    importances = dt.feature_importances_
    print(f"\n  Feature importance (Decision Tree):", flush=True)
    for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp * 40)
        print(f"    {name:<25} {imp:.4f}  {bar}", flush=True)

    # ── PLOTS ──
    SAVE_DIR = os.path.join(os.getcwd(), "results_metaclassifier")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Plot 1: Method comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    method_names = ["SelfCheckGPT\n(single)", "Equal-weight\n(3 signals)", "Logistic Reg\n(learned)", "Neural Net\n(MLP)"]
    method_aurocs = [auroc_nli_all, auroc_equal_all, auroc_lr_all, auroc_mlp_all]
    bar_colors = ["#f59e0b", "#94a3b8", "#3b82f6", "#8b5cf6"]

    bars = axes[0].bar(method_names, method_aurocs, color=bar_colors, edgecolor="white", width=0.6)
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("Unified Meta-Classifier Comparison", fontweight="bold")
    axes[0].set_ylim(min(method_aurocs) - 0.03, max(method_aurocs) + 0.03)
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    for bar, val in zip(bars, method_aurocs):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.003,
                     f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")

    # Plot 2: Signal importance
    sorted_idx = np.argsort(clf_all.coef_[0])
    sorted_names = [feat_names[i] for i in sorted_idx]
    sorted_coefs = clf_all.coef_[0][sorted_idx]
    bar_colors2 = ["#ef4444" if c < 0 else "#22c55e" for c in sorted_coefs]
    axes[1].barh(sorted_names, sorted_coefs, color=bar_colors2, edgecolor="white")
    axes[1].set_xlabel("Learned Weight (Logistic Regression)")
    axes[1].set_title("Signal Importance for Detection", fontweight="bold")
    axes[1].axvline(0, color="gray", linewidth=0.8)

    plt.suptitle(f"Learned Meta-Classifier (n={len(y_all)} samples, 5-fold CV)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "metaclassifier_comparison.png"), bbox_inches="tight", dpi=150)
    print(f"\n  Saved metaclassifier_comparison.png", flush=True)

    # Plot 3: Per-source comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    src_names = list(per_source.keys())
    eq_aurocs = [per_source[s]["auroc_equal"] for s in src_names]
    lr_aurocs = [per_source[s]["auroc_lr"] for s in src_names]
    mlp_aurocs = [per_source[s]["auroc_mlp"] for s in src_names]
    x = np.arange(len(src_names))
    width = 0.25
    ax.bar(x - width, eq_aurocs, width, label="Equal-weight", color="#94a3b8", edgecolor="white")
    ax.bar(x, lr_aurocs, width, label="Logistic Reg", color="#3b82f6", edgecolor="white")
    ax.bar(x + width, mlp_aurocs, width, label="MLP", color="#8b5cf6", edgecolor="white")
    ax.set_ylabel("AUROC")
    ax.set_title("Per-Source: Equal vs Learned Classifiers", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("-", "\n") for s in src_names], fontsize=8)
    ax.legend()
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "persource_comparison.png"), bbox_inches="tight", dpi=150)
    print(f"  Saved persource_comparison.png", flush=True)

    # Save results
    with open(os.path.join(SAVE_DIR, "metaclassifier_results.pkl"), "wb") as f:
        pickle.dump({
            "per_source": per_source,
            "unified_auroc_lr": auroc_lr_all,
            "unified_auroc_mlp": auroc_mlp_all,
            "unified_auroc_equal": auroc_equal_all,
            "learned_weights": dict(zip(feat_names, clf_all.coef_[0].tolist())),
            "feature_importance_dt": dict(zip(feat_names, importances.tolist())),
            "total_samples": len(y_all),
        }, f)
    print(f"  Saved metaclassifier_results.pkl", flush=True)

    # ── SUMMARY ──
    print(f"\n{'='*70}", flush=True)
    print("  SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    best_method = max(zip(method_names, method_aurocs), key=lambda x: x[1])
    print(f"  Best method: {best_method[0].replace(chr(10), ' ')} (AUROC={best_method[1]:.4f})", flush=True)
    print(f"  Improvement over equal-weight: {best_method[1] - auroc_equal_all:+.4f}", flush=True)
    print(f"  Most important signal: {feat_names[np.argmax(np.abs(clf_all.coef_[0]))]}", flush=True)
    print(f"{'='*70}", flush=True)

else:
    print("Need at least 2 data sources to train unified classifier.", flush=True)

print("\nDone!", flush=True)
input("Press Enter to exit...")