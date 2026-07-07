import os
import pickle
import traceback
import numpy as np
from collections import Counter

print("Loading results...", flush=True)
SAVE_DIR = os.path.join(os.getcwd(), "results")

with open(os.path.join(SAVE_DIR, "final_results.pkl"), "rb") as f:
    data = pickle.load(f)

results = data["results"]
print(f"Loaded {len(results)} questions", flush=True)

try:
    # ============================================================
    #  STEP 1: COMPUTE THRESHOLDS
    # ============================================================
    sc_vals = np.array([r["uncertainty_sc"] for r in results])
    se_vals = np.array([r["uncertainty_se"] for r in results])
    ae_vals = np.array([r["uncertainty_attn"] for r in results])
    lp_vals = np.array([r["avg_neg_log_prob"] for r in results])

    sc_med = np.median(sc_vals)
    se_med = np.median(se_vals)
    ae_med = np.median(ae_vals)
    lp_med = np.median(lp_vals)

    print(f"\nSignal medians (high/low threshold):", flush=True)
    print(f"  Self-Consistency:  {sc_med:.4f}", flush=True)
    print(f"  Semantic Entropy:  {se_med:.4f}", flush=True)
    print(f"  Attention Entropy: {ae_med:.4f}", flush=True)
    print(f"  Avg(-logP):        {lp_med:.4f}", flush=True)

    # ============================================================
    #  STEP 2: CLASSIFY EACH QUESTION INTO A HALLUCINATION TYPE
    # ============================================================
    def classify_fingerprint(r):
        sc_high = r["uncertainty_sc"] > sc_med
        se_high = r["uncertainty_se"] > se_med
        lp_high = r["avg_neg_log_prob"] > lp_med
        is_wrong = not r["is_correct"]

        if not is_wrong:
            if not sc_high and not se_high:
                return "correct_confident"
            else:
                return "correct_uncertain"

        if not sc_high and not se_high:
            return "confident_fabrication"
        elif sc_high and se_high:
            return "knowledge_gap"
        elif not sc_high and se_high:
            return "shallow_mimicry"
        elif sc_high and not se_high:
            return "inconsistent_surface"
        else:
            return "mixed_signals"

    for r in results:
        r["fingerprint_type"] = classify_fingerprint(r)

    type_counts = Counter(r["fingerprint_type"] for r in results)

    print(f"\n{'='*60}", flush=True)
    print(f"  HALLUCINATION FINGERPRINT DISTRIBUTION", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\n{'Type':<25} {'Count':<8} {'% of Total':<12}", flush=True)
    print("-" * 45, flush=True)
    for ftype, count in type_counts.most_common():
        pct = count / len(results) * 100
        print(f"{ftype:<25} {count:<8} {pct:<12.1f}%", flush=True)

    # ============================================================
    #  STEP 3: DANGER ANALYSIS
    # ============================================================
    print(f"\n{'='*65}", flush=True)
    print(f"  DANGER ANALYSIS: Hallucination rate per type", flush=True)
    print(f"{'='*65}", flush=True)

    fingerprint_types = sorted(set(r["fingerprint_type"] for r in results))

    print(f"\n{'Type':<25} {'N':<6} {'Hall%':<8} "
          f"{'AvgSC':<9} {'AvgSE':<9} {'AvgLP':<9}", flush=True)
    print("-" * 66, flush=True)

    type_analysis = {}
    for ftype in fingerprint_types:
        sub = [r for r in results if r["fingerprint_type"] == ftype]
        n = len(sub)
        if n == 0:
            continue
        hall_rate = sum(1 for r in sub if not r["is_correct"]) / n
        avg_sc = np.mean([r["uncertainty_sc"] for r in sub])
        avg_se = np.mean([r["uncertainty_se"] for r in sub])
        avg_lp = np.mean([r["avg_neg_log_prob"] for r in sub])

        type_analysis[ftype] = {
            "n": n, "hallucination_rate": hall_rate,
            "avg_sc": avg_sc, "avg_se": avg_se, "avg_lp": avg_lp,
        }
        print(f"{ftype:<25} {n:<6} {hall_rate:<8.1%} "
              f"{avg_sc:<9.3f} {avg_se:<9.3f} {avg_lp:<9.3f}", flush=True)

    if "confident_fabrication" in type_analysis:
        cf = type_analysis["confident_fabrication"]
        print(f"\n** KEY FINDING **", flush=True)
        print(f"   'Confident fabrication': {cf['hallucination_rate']:.0%} hallucination rate", flush=True)
        print(f"   but all uncertainty signals are LOW - standard detection MISSES these!", flush=True)
        print(f"   This affects {cf['n']} questions ({cf['n']/len(results)*100:.1f}% of dataset)", flush=True)

    # ============================================================
    #  STEP 4: FINGERPRINT-ENHANCED SCORING
    # ============================================================
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    labels = np.array([float(not r["is_correct"]) for r in results])

    def mnorm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)

    # Method 1: Equal-weight (from main pipeline)
    combined_equal = (mnorm(sc_vals) + mnorm(se_vals) + mnorm(lp_vals)) / 3.0

    # Method 2: Fingerprint-enhanced dynamic weighting
    enhanced = []
    for r in results:
        sc_n = (r["uncertainty_sc"] - sc_vals.min()) / (sc_vals.max() - sc_vals.min() + 1e-12)
        se_n = (r["uncertainty_se"] - se_vals.min()) / (se_vals.max() - se_vals.min() + 1e-12)
        lp_n = (r["avg_neg_log_prob"] - lp_vals.min()) / (lp_vals.max() - lp_vals.min() + 1e-12)

        if sc_n < 0.3:
            # Low self-consistency uncertainty = could be confident fabrication
            # Upweight other signals
            score = 0.1 * sc_n + 0.45 * se_n + 0.45 * lp_n
        else:
            score = 0.35 * sc_n + 0.35 * se_n + 0.30 * lp_n
        enhanced.append(score)
    enhanced = np.array(enhanced)

    # Method 3: Learned meta-classifier
    features = np.column_stack([mnorm(sc_vals), mnorm(se_vals), mnorm(lp_vals)])
    clf = LogisticRegression(random_state=42, max_iter=1000)
    learned = cross_val_predict(clf, features, labels, cv=5, method="predict_proba")[:, 1]

    # Fit on full data to see weights
    clf.fit(features, labels)
    feat_names = ["Self-Consistency", "Semantic Entropy", "Avg(-logP)"]

    print(f"\n{'='*60}", flush=True)
    print(f"  LEARNED SIGNAL WEIGHTS (Logistic Regression)", flush=True)
    print(f"{'='*60}", flush=True)
    for name, coef in zip(feat_names, clf.coef_[0]):
        bar = "#" * int(abs(coef) * 5)
        print(f"  {name:<22} {coef:+.3f}  {bar}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  DETECTION COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)

    # Individual methods
    selfcheck_score = np.array([r["uncertainty_selfcheck"] for r in results])

    print(f"\n{'Method':<35} {'AUROC':<9} {'AUPRC':<9}", flush=True)
    print("-" * 53, flush=True)

    all_methods = [
        ("Self-Consistency", sc_vals),
        ("Semantic Entropy", se_vals),
        ("SelfCheckGPT-NLI (baseline)", selfcheck_score),
        ("Equal-weight combined", combined_equal),
        ("Fingerprint-enhanced", enhanced),
        ("Learned meta-classifier (CV)", learned),
    ]

    method_results = {}
    for name, sc in all_methods:
        auroc = roc_auc_score(labels, sc)
        auprc = average_precision_score(labels, sc)
        method_results[name] = {"auroc": auroc, "auprc": auprc}
        print(f"{name:<35} {auroc:<9.4f} {auprc:<9.4f}", flush=True)

    # ============================================================
    #  STEP 5: ADAPTIVE RESPONSE POLICY
    # ============================================================
    print(f"\n{'='*65}", flush=True)
    print(f"  ADAPTIVE RESPONSE POLICY (fingerprint-aware)", flush=True)
    print(f"{'='*65}", flush=True)

    strategies = {
        "correct_confident": "Answer directly",
        "correct_uncertain": "Answer with uncertainty flag",
        "knowledge_gap": "ABSTAIN - model clearly guessing",
        "confident_fabrication": "FLAG for verification - wrong but looks confident",
        "shallow_mimicry": "FLAG - surface match but meaning differs",
        "inconsistent_surface": "Answer with uncertainty flag",
        "mixed_signals": "Answer with uncertainty flag",
    }

    print(f"\n{'Type':<25} {'Strategy':<45} {'N':<6}", flush=True)
    print("-" * 76, flush=True)
    for ftype in type_counts.most_common():
        strategy = strategies.get(ftype[0], "Unknown")
        print(f"{ftype[0]:<25} {strategy:<45} {ftype[1]:<6}", flush=True)

    # Simulate policy
    answered = 0
    correct_answered = 0
    abstained = 0
    flagged = 0

    for r in results:
        ft = r["fingerprint_type"]
        if ft in ("correct_confident",):
            answered += 1
            if r["is_correct"]:
                correct_answered += 1
        elif ft in ("correct_uncertain", "inconsistent_surface", "mixed_signals"):
            answered += 1
            flagged += 1
            if r["is_correct"]:
                correct_answered += 1
        elif ft in ("knowledge_gap",):
            abstained += 1
        elif ft in ("confident_fabrication", "shallow_mimicry"):
            answered += 1
            flagged += 1
            if r["is_correct"]:
                correct_answered += 1

    baseline_acc = sum(r["is_correct"] for r in results) / len(results)
    policy_acc = correct_answered / answered if answered > 0 else 0
    coverage = answered / len(results)

    print(f"\n  Baseline (answer everything):  {baseline_acc:.2%} accuracy", flush=True)
    print(f"  Fingerprint-aware policy:", flush=True)
    print(f"    Accuracy on answered:        {policy_acc:.2%}", flush=True)
    print(f"    Coverage:                    {coverage:.1%}", flush=True)
    print(f"    Questions abstained:         {abstained}", flush=True)
    print(f"    Questions flagged:           {flagged}", flush=True)
    print(f"    Accuracy improvement:        {policy_acc - baseline_acc:+.2%}", flush=True)

    # ============================================================
    #  STEP 6: PLOTS
    # ============================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot 1: Fingerprint distribution + hallucination rates
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    types_sorted = [t[0] for t in type_counts.most_common()]
    counts_sorted = [t[1] for t in type_counts.most_common()]
    rates_sorted = [type_analysis.get(t, {}).get("hallucination_rate", 0) for t in types_sorted]

    type_colors = {
        "correct_confident": "#97C459",
        "correct_uncertain": "#C0DD97",
        "knowledge_gap": "#85B7EB",
        "confident_fabrication": "#E24B4A",
        "shallow_mimicry": "#EF9F27",
        "inconsistent_surface": "#F0997B",
        "mixed_signals": "#B4B2A9",
    }
    colors = [type_colors.get(t, "#B4B2A9") for t in types_sorted]

    axes[0].barh(types_sorted, counts_sorted, color=colors, edgecolor="white")
    axes[0].set_xlabel("Number of questions")
    axes[0].set_title("Fingerprint type distribution")

    axes[1].barh(types_sorted, rates_sorted, color=colors, edgecolor="white")
    axes[1].set_xlabel("Hallucination rate")
    axes[1].set_title("Hallucination rate by type")
    for i, rate in enumerate(rates_sorted):
        axes[1].text(rate + 0.02, i, f"{rate:.0%}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "fingerprint_distribution.png"), bbox_inches="tight")
    print("\nSaved fingerprint_distribution.png", flush=True)

    # Plot 2: Signal heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    hall_types = [ft for ft in types_sorted if ft in type_analysis]
    sig_names = ["Self-Consistency", "Semantic Entropy", "Avg(-logP)"]
    hdata = []
    for ft in hall_types:
        ta = type_analysis[ft]
        hdata.append([ta["avg_sc"], ta["avg_se"], ta["avg_lp"]])
    hdata = np.array(hdata)
    # Normalize columns
    for col in range(hdata.shape[1]):
        cmin, cmax = hdata[:, col].min(), hdata[:, col].max()
        if cmax - cmin > 1e-10:
            hdata[:, col] = (hdata[:, col] - cmin) / (cmax - cmin)
        else:
            hdata[:, col] = 0.5
    im = ax.imshow(hdata, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(sig_names)))
    ax.set_xticklabels(sig_names, fontsize=10)
    ax.set_yticks(range(len(hall_types)))
    ax.set_yticklabels(hall_types, fontsize=10)
    ax.set_title("Signal fingerprint heatmap (red = high uncertainty)")
    for i in range(len(hall_types)):
        for j in range(len(sig_names)):
            color = "white" if hdata[i, j] > 0.6 or hdata[i, j] < 0.4 else "black"
            ax.text(j, i, f"{hdata[i,j]:.2f}", ha="center", va="center", color=color, fontsize=10)
    plt.colorbar(im, ax=ax, label="Normalized uncertainty")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "fingerprint_heatmap.png"), bbox_inches="tight")
    print("Saved fingerprint_heatmap.png", flush=True)

    # Plot 3: Method comparison
    fig, ax = plt.subplots(figsize=(9, 4))
    mnames = ["Equal-weight\ncombined", "Fingerprint-\nenhanced", "Learned meta-\nclassifier"]
    maurocs = [
        method_results["Equal-weight combined"]["auroc"],
        method_results["Fingerprint-enhanced"]["auroc"],
        method_results["Learned meta-classifier (CV)"]["auroc"],
    ]
    bcolors = ["#B4B2A9", "#5DCAA5", "#AFA9EC"]
    bars = ax.bar(mnames, maurocs, color=bcolors, edgecolor="white", width=0.5)
    ax.set_ylabel("AUROC")
    ax.set_title("Novel contribution: fingerprint-aware vs standard")
    ax.set_ylim(min(maurocs) - 0.03, max(maurocs) + 0.03)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars, maurocs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "novel_contribution.png"), bbox_inches="tight")
    print("Saved novel_contribution.png", flush=True)

    # ============================================================
    #  FINAL SUMMARY
    # ============================================================
    print(f"\n{'='*70}", flush=True)
    print(f"  COMPLETE THESIS RESULTS (WITH NOVEL CONTRIBUTION)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\n  PART 1: Uncertainty Methods", flush=True)
    for name, res in method_results.items():
        print(f"    {name:<35} AUROC: {res['auroc']:.4f}", flush=True)

    print(f"\n  PART 2: Hallucination Taxonomy", flush=True)
    for ft in ["confident_fabrication", "knowledge_gap", "shallow_mimicry"]:
        if ft in type_analysis:
            ta = type_analysis[ft]
            print(f"    {ft:<25} n={ta['n']:<4} hall_rate={ta['hallucination_rate']:.0%}", flush=True)

    print(f"\n  PART 3: Adaptive Policy", flush=True)
    print(f"    Baseline accuracy: {baseline_acc:.2%}", flush=True)
    print(f"    Policy accuracy:   {policy_acc:.2%} (coverage: {coverage:.1%})", flush=True)
    print(f"    Improvement:       {policy_acc - baseline_acc:+.2%}", flush=True)
    print(f"{'='*70}", flush=True)

    # Save everything
    with open(os.path.join(SAVE_DIR, "fingerprint_results.pkl"), "wb") as f:
        pickle.dump({
            "results": results,
            "type_analysis": type_analysis,
            "type_counts": dict(type_counts),
            "method_results": method_results,
            "learned_weights": dict(zip(feat_names, clf.coef_[0].tolist())),
            "policy": {"accuracy": policy_acc, "coverage": coverage,
                       "abstained": abstained, "flagged": flagged},
        }, f)
    print(f"\nSaved fingerprint_results.pkl", flush=True)

except Exception as e:
    print(f"\nERROR: {e}", flush=True)
    traceback.print_exc()

print("\nALL DONE!", flush=True)
print(f"Results in: {SAVE_DIR}", flush=True)
input("\nPress Enter to exit...")
