import os
import pickle
import traceback
import numpy as np
from collections import Counter

print("=== RAG VERIFICATION FOR HALLUCINATION DETECTION ===", flush=True)
print("This adds evidence retrieval to catch confident fabrications", flush=True)

SAVE_DIR = os.path.join(os.getcwd(), "results")

# Load existing results
print("\nLoading pipeline results...", flush=True)
with open(os.path.join(SAVE_DIR, "final_results.pkl"), "rb") as f:
    data = pickle.load(f)
results = data["results"]
print(f"Loaded {len(results)} questions", flush=True)

# Install wikipedia package if needed
try:
    import wikipedia
except ImportError:
    print("Installing wikipedia package...", flush=True)
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wikipedia", "-q"])
    import wikipedia

# Load NLI model for evidence verification
print("Loading NLI model...", flush=True)
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLI_MODEL = "microsoft/deberta-large-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
try:
    nli_model = nli_model.to("cuda")
    nli_device = "cuda"
except:
    nli_device = "cpu"
print(f"NLI model on {nli_device}", flush=True)


def retrieve_evidence(question, max_sentences=5):
    """Retrieve relevant evidence from Wikipedia for a question."""
    try:
        # Search Wikipedia for relevant pages
        search_results = wikipedia.search(question, results=3)
        if not search_results:
            return None, []

        evidence_passages = []
        source_titles = []

        for title in search_results[:2]:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                # Get first few sentences of the page
                sentences = page.content.split(". ")[:max_sentences]
                passage = ". ".join(sentences) + "."
                evidence_passages.append(passage)
                source_titles.append(title)
            except (wikipedia.exceptions.DisambiguationError,
                    wikipedia.exceptions.PageError):
                continue

        if evidence_passages:
            combined = " ".join(evidence_passages)
            return combined, source_titles
        return None, []

    except Exception as e:
        return None, []


def verify_answer_with_evidence(answer, evidence):
    """Use NLI to check if evidence supports or contradicts the answer.

    Returns:
        support_score: probability that evidence SUPPORTS the answer
        contradict_score: probability that evidence CONTRADICTS the answer
        verdict: 'supported', 'contradicted', or 'neutral'
    """
    if not answer.strip() or not evidence:
        return 0.5, 0.5, "no_evidence"

    # Truncate evidence to fit model's max length
    max_evidence_len = 400  # characters
    evidence_truncated = evidence[:max_evidence_len]

    # NLI: premise = evidence, hypothesis = "The answer is [answer]"
    premise = evidence_truncated
    hypothesis = f"The answer to the question is: {answer}"

    inputs = nli_tokenizer(
        premise, hypothesis,
        return_tensors="pt", truncation=True, max_length=512
    ).to(nli_device)

    with torch.no_grad():
        logits = nli_model(**inputs).logits

    # DeBERTa-MNLI: 0=contradiction, 1=neutral, 2=entailment
    probs = torch.softmax(logits[0], dim=-1)
    contradict_prob = probs[0].item()
    neutral_prob = probs[1].item()
    support_prob = probs[2].item()

    if support_prob > 0.5:
        verdict = "supported"
    elif contradict_prob > 0.5:
        verdict = "contradicted"
    else:
        verdict = "neutral"

    return support_prob, contradict_prob, verdict


# ── MAIN: RUN RAG VERIFICATION ON ALL QUESTIONS ──
print("\nRetrieving Wikipedia evidence and verifying answers...", flush=True)
print("This takes about 5-10 minutes (API calls)\n", flush=True)

from tqdm import tqdm
import time

for i in tqdm(range(len(results)), desc="RAG Verification"):
    r = results[i]
    question = r["question"]
    answer = r["greedy_response"]

    # Retrieve evidence
    evidence, sources = retrieve_evidence(question)
    r["rag_evidence"] = evidence
    r["rag_sources"] = sources

    if evidence:
        support, contradict, verdict = verify_answer_with_evidence(answer, evidence)
        r["rag_support_score"] = support
        r["rag_contradict_score"] = contradict
        r["rag_verdict"] = verdict
    else:
        r["rag_support_score"] = 0.5
        r["rag_contradict_score"] = 0.5
        r["rag_verdict"] = "no_evidence"

    # Small delay to avoid hitting Wikipedia rate limits
    time.sleep(0.5)

    if (i + 1) % 25 == 0:
        with open(os.path.join(SAVE_DIR, "rag_checkpoint.pkl"), "wb") as f:
            pickle.dump(results, f)

print("\nRAG verification complete!", flush=True)

# ── ANALYSIS ──
print("\n" + "=" * 70, flush=True)
print("  RAG VERIFICATION RESULTS", flush=True)
print("=" * 70, flush=True)

# Overall verdict distribution
verdicts = Counter(r["rag_verdict"] for r in results)
print(f"\nVerdict distribution:", flush=True)
for v, c in verdicts.most_common():
    print(f"  {v:<20} {c:>4} ({c/len(results)*100:.1f}%)", flush=True)

# RAG accuracy: does RAG verdict match actual correctness?
print(f"\nRAG verdict vs actual correctness:", flush=True)
print(f"{'Verdict':<20} {'Correct':<10} {'Wrong':<10} {'Precision':<10}", flush=True)
print("-" * 50, flush=True)

for verdict in ["supported", "contradicted", "neutral", "no_evidence"]:
    sub = [r for r in results if r["rag_verdict"] == verdict]
    if not sub:
        continue
    n_correct = sum(1 for r in sub if r["is_correct"])
    n_wrong = len(sub) - n_correct
    precision = n_correct / len(sub) if verdict == "supported" else n_wrong / len(sub)
    label = "support precision" if verdict == "supported" else "contradict precision"
    print(f"{verdict:<20} {n_correct:<10} {n_wrong:<10} {precision:<10.2%}", flush=True)

# KEY: Does RAG catch confident fabrications?
print(f"\n{'='*70}", flush=True)
print("  KEY FINDING: Does RAG catch confident fabrications?", flush=True)
print(f"{'='*70}", flush=True)

confident_fabs = [r for r in results if r.get("fingerprint_type") == "confident_fabrication"]
print(f"\nConfident fabrications found: {len(confident_fabs)}", flush=True)

if confident_fabs:
    for r in confident_fabs:
        q = r["question"][:60]
        ans = r["greedy_response"][:40]
        verdict = r["rag_verdict"]
        support = r["rag_support_score"]
        contradict = r["rag_contradict_score"]
        print(f"\n  Q: {q}...", flush=True)
        print(f"  A: {ans}...", flush=True)
        print(f"  RAG verdict: {verdict} (support={support:.3f}, contradict={contradict:.3f})", flush=True)
        if r["rag_sources"]:
            print(f"  Sources: {', '.join(r['rag_sources'][:2])}", flush=True)
        caught = verdict in ("contradicted", "neutral")
        print(f"  Caught by RAG: {'YES' if caught else 'NO'}", flush=True)

    caught_count = sum(1 for r in confident_fabs
                       if r["rag_verdict"] in ("contradicted", "neutral"))
    print(f"\n  RAG caught {caught_count}/{len(confident_fabs)} confident fabrications "
          f"({caught_count/len(confident_fabs)*100:.0f}%)", flush=True)
    print(f"  (These are hallucinations that ALL uncertainty signals missed!)", flush=True)

# Combined system: uncertainty signals + RAG
print(f"\n{'='*70}", flush=True)
print("  COMBINED SYSTEM: Uncertainty + RAG", flush=True)
print(f"{'='*70}", flush=True)

# New policy: use fingerprint + RAG verdict
answered = 0
correct_answered = 0
abstained = 0
flagged = 0
rag_caught = 0

for r in results:
    ft = r.get("fingerprint_type", "unknown")
    rag_v = r.get("rag_verdict", "no_evidence")

    # Decision logic:
    # 1. Knowledge gap -> abstain (uncertainty catches these)
    # 2. RAG contradicts answer -> flag (RAG catches confident fabrications)
    # 3. RAG supports + low uncertainty -> answer directly
    # 4. Everything else -> answer with flag

    if ft == "knowledge_gap":
        abstained += 1
    elif rag_v == "contradicted":
        # RAG says answer is wrong
        flagged += 1
        rag_caught += 1
        answered += 1
        if r["is_correct"]:
            correct_answered += 1
    elif rag_v == "supported" and ft == "correct_confident":
        # Both uncertainty and RAG agree: confident and supported
        answered += 1
        if r["is_correct"]:
            correct_answered += 1
    else:
        # Answer with flag
        answered += 1
        flagged += 1
        if r["is_correct"]:
            correct_answered += 1

baseline_acc = sum(r["is_correct"] for r in results) / len(results)
uncertainty_only_acc = 0.9387  # from fingerprinting results
combined_acc = correct_answered / answered if answered > 0 else 0
coverage = answered / len(results)

print(f"\n  {'System':<35} {'Accuracy':<12} {'Coverage':<10}", flush=True)
print("  " + "-" * 57, flush=True)
print(f"  {'No intervention (baseline)':<35} {baseline_acc:<12.2%} {'100.0%':<10}", flush=True)
print(f"  {'Uncertainty only (fingerprint)':<35} {uncertainty_only_acc:<12.2%} {'81.5%':<10}", flush=True)
print(f"  {'Uncertainty + RAG (combined)':<35} {combined_acc:<12.2%} {coverage:<10.1%}", flush=True)
print(f"\n  RAG additionally caught {rag_caught} answers that uncertainty missed", flush=True)

# Save final results
with open(os.path.join(SAVE_DIR, "rag_results.pkl"), "wb") as f:
    pickle.dump({
        "results": results,
        "verdicts": dict(verdicts),
        "baseline_acc": baseline_acc,
        "combined_acc": combined_acc,
        "coverage": coverage,
        "confident_fabs_caught": caught_count if confident_fabs else 0,
    }, f)
print(f"\nSaved rag_results.pkl", flush=True)

# Generate plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: System comparison
systems = ["Baseline\n(no intervention)", "Uncertainty\n(fingerprint)", "Uncertainty\n+ RAG"]
accs = [baseline_acc * 100, uncertainty_only_acc * 100, combined_acc * 100]
colors = ["#ef4444", "#f59e0b", "#22c55e"]
bars = axes[0].bar(systems, accs, color=colors, edgecolor="white", width=0.5)
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("System Comparison: Accuracy", fontsize=13, fontweight="bold")
axes[0].set_ylim(70, 100)
for bar, val in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.5,
                 f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

# Plot 2: RAG verdict distribution for correct vs wrong
correct_support = sum(1 for r in results if r["is_correct"] and r["rag_verdict"] == "supported")
correct_contradict = sum(1 for r in results if r["is_correct"] and r["rag_verdict"] == "contradicted")
correct_neutral = sum(1 for r in results if r["is_correct"] and r["rag_verdict"] in ("neutral", "no_evidence"))
wrong_support = sum(1 for r in results if not r["is_correct"] and r["rag_verdict"] == "supported")
wrong_contradict = sum(1 for r in results if not r["is_correct"] and r["rag_verdict"] == "contradicted")
wrong_neutral = sum(1 for r in results if not r["is_correct"] and r["rag_verdict"] in ("neutral", "no_evidence"))

x = np.arange(3)
width = 0.35
axes[1].bar(x - width/2, [correct_support, correct_contradict, correct_neutral],
            width, label="Correct answers", color="#22c55e", edgecolor="white")
axes[1].bar(x + width/2, [wrong_support, wrong_contradict, wrong_neutral],
            width, label="Hallucinated answers", color="#ef4444", edgecolor="white")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["Supported", "Contradicted", "Neutral/None"])
axes[1].set_ylabel("Count")
axes[1].set_title("RAG Verdict: Correct vs Hallucinated", fontsize=13, fontweight="bold")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "rag_comparison.png"), bbox_inches="tight", dpi=150)
print("Saved rag_comparison.png", flush=True)

print(f"\n{'='*70}", flush=True)
print("  DONE! RAG verification complete.", flush=True)
print(f"{'='*70}", flush=True)

# Cleanup
del nli_model, nli_tokenizer
torch.cuda.empty_cache()

input("\nPress Enter to exit...")
