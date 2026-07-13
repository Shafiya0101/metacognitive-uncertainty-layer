import os, pickle, time, traceback
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

print("=" * 60, flush=True)
print("  ATTENTION ENTROPY REVISITED", flush=True)
print("  Testing early, middle, and last layers", flush=True)
print("=" * 60, flush=True)

# Load model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
print(f"\nLoading {MODEL_NAME}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=qc,
                                              device_map="auto", dtype=torch.float16)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
print("Model loaded!", flush=True)

n_layers = model.config.num_hidden_layers
print(f"Total layers: {n_layers}", flush=True)

# Load existing results for questions and labels
with open(os.path.join(os.getcwd(), "results", "final_results.pkl"), "rb") as f:
    data = pickle.load(f)
results = data["results"]
print(f"Loaded {len(results)} questions", flush=True)

def format_prompt(question):
    msgs = [{"role": "user", "content": "Answer the following question with a short, direct answer. Do not explain.\n\nQuestion: " + question + "\nAnswer:"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def compute_attention_entropy_per_layer(question):
    """Compute attention entropy for EACH layer during generation."""
    prompt = format_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=50, do_sample=False,
            output_attentions=True, return_dict_in_generate=True,
        )

    # Collect per-layer entropies
    layer_entropies = {i: [] for i in range(n_layers)}

    if hasattr(outputs, "attentions") and outputs.attentions:
        for step_attn in outputs.attentions:
            if step_attn is None or len(step_attn) == 0:
                continue
            for layer_idx, layer_attn in enumerate(step_attn):
                if layer_attn is None:
                    continue
                # layer_attn shape: (batch, heads, 1, seq_len)
                aw = layer_attn[0, :, 0, :].float()
                h = -(aw * torch.log(aw + 1e-12)).sum(-1).mean().item()
                layer_entropies[layer_idx].append(h)

    # Average entropy per layer
    avg_per_layer = {}
    for layer_idx in range(n_layers):
        if layer_entropies[layer_idx]:
            avg_per_layer[layer_idx] = float(np.mean(layer_entropies[layer_idx]))
        else:
            avg_per_layer[layer_idx] = 0.0

    return avg_per_layer

# ── RUN ATTENTION ANALYSIS ──
print(f"\nExtracting attention from all {n_layers} layers...", flush=True)
print("This takes about 15-20 minutes", flush=True)

SAVE_DIR = os.path.join(os.getcwd(), "results_attention")
os.makedirs(SAVE_DIR, exist_ok=True)

# Check checkpoint
ckpt_path = os.path.join(SAVE_DIR, "attention_layers_ckpt.pkl")
if os.path.exists(ckpt_path):
    with open(ckpt_path, "rb") as f:
        layer_data = pickle.load(f)
    start_idx = len(layer_data)
    print(f"Resuming from question {start_idx}", flush=True)
else:
    layer_data = []
    start_idx = 0

t0 = time.time()
for i in tqdm(range(start_idx, len(results)), desc="Attention analysis"):
    try:
        per_layer = compute_attention_entropy_per_layer(results[i]["question"])
        layer_data.append(per_layer)
    except Exception as e:
        print(f"  Error at q{i}: {e}", flush=True)
        layer_data.append({j: 0.0 for j in range(n_layers)})

    if (i + 1) % 25 == 0:
        with open(ckpt_path, "wb") as f:
            pickle.dump(layer_data, f)

with open(ckpt_path, "wb") as f:
    pickle.dump(layer_data, f)
print(f"Done in {(time.time()-t0)/60:.1f} min", flush=True)

# ── ANALYSIS ──
labels = np.array([float(not r["is_correct"]) for r in results])

# Group layers
layer_groups = {
    "Early (0-7)": list(range(0, min(8, n_layers))),
    "Middle (8-15)": list(range(8, min(16, n_layers))),
    "Late (16-23)": list(range(16, min(24, n_layers))),
    "Final (24-31)": list(range(24, n_layers)),
    "Last layer only": [n_layers - 1],
    "All layers avg": list(range(n_layers)),
    "Rollout (weighted)": "rollout",
}

print(f"\n{'='*60}", flush=True)
print("  ATTENTION ENTROPY BY LAYER GROUP", flush=True)
print(f"{'='*60}", flush=True)
print(f"\n{'Layer Group':<25} {'AUROC':<10} {'Avg Entropy':<12}", flush=True)
print("-" * 47, flush=True)

group_aurocs = {}
for group_name, layer_indices in layer_groups.items():
    if layer_indices == "rollout":
        # Attention rollout: product of attention across layers (simplified as weighted avg)
        scores = []
        for ld in layer_data:
            all_vals = [ld.get(j, 0) for j in range(n_layers)]
            # Weighted: later layers get more weight
            weights = np.linspace(0.5, 1.5, n_layers)
            scores.append(float(np.average(all_vals, weights=weights)))
    else:
        scores = []
        for ld in layer_data:
            vals = [ld.get(j, 0) for j in layer_indices]
            scores.append(float(np.mean(vals)) if vals else 0.0)

    scores = np.array(scores)
    if np.std(scores) > 1e-10 and np.std(labels) > 0:
        auroc = roc_auc_score(labels, scores)
    else:
        auroc = 0.5

    group_aurocs[group_name] = auroc
    avg_ent = np.mean(scores)
    print(f"{group_name:<25} {auroc:<10.4f} {avg_ent:<12.4f}", flush=True)

# Per-layer AUROC
print(f"\n{'='*60}", flush=True)
print("  PER-LAYER AUROC (all layers)", flush=True)
print(f"{'='*60}", flush=True)

per_layer_auroc = []
for layer_idx in range(n_layers):
    scores = np.array([ld.get(layer_idx, 0) for ld in layer_data])
    if np.std(scores) > 1e-10:
        auroc = roc_auc_score(labels, scores)
    else:
        auroc = 0.5
    per_layer_auroc.append(auroc)

# Show best and worst layers
sorted_layers = sorted(range(n_layers), key=lambda x: per_layer_auroc[x], reverse=True)
print(f"\nTop 5 layers:", flush=True)
for idx in sorted_layers[:5]:
    print(f"  Layer {idx}: AUROC = {per_layer_auroc[idx]:.4f}", flush=True)
print(f"\nBottom 5 layers:", flush=True)
for idx in sorted_layers[-5:]:
    print(f"  Layer {idx}: AUROC = {per_layer_auroc[idx]:.4f}", flush=True)

# Best single layer vs best group
best_layer = sorted_layers[0]
best_group = max(group_aurocs, key=group_aurocs.get)
print(f"\nBest single layer: {best_layer} (AUROC = {per_layer_auroc[best_layer]:.4f})", flush=True)
print(f"Best group: {best_group} (AUROC = {group_aurocs[best_group]:.4f})", flush=True)
print(f"Original (last layer only): AUROC = {per_layer_auroc[-1]:.4f}", flush=True)

# Compare with other signals
print(f"\nFor comparison:", flush=True)
print(f"  Self-Consistency AUROC:  0.7683", flush=True)
print(f"  Semantic Entropy AUROC:  0.7729", flush=True)
print(f"  SelfCheckGPT-NLI AUROC:  0.7924", flush=True)
print(f"  Best attention AUROC:    {max(per_layer_auroc):.4f}", flush=True)

# ── PLOT ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Per-layer AUROC
axes[0].bar(range(n_layers), per_layer_auroc, color="#3b82f6", edgecolor="white", width=0.8)
axes[0].axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random (0.5)")
axes[0].axhline(0.7924, color="#22c55e", linestyle="--", linewidth=1, label="SelfCheckGPT (0.79)")
axes[0].set_xlabel("Layer Index")
axes[0].set_ylabel("AUROC")
axes[0].set_title("Attention Entropy AUROC by Layer", fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", alpha=0.3)

# Plot 2: Group comparison
gnames = list(group_aurocs.keys())
gvals = [group_aurocs[g] for g in gnames]
gcolors = ["#22c55e" if v > 0.55 else "#f59e0b" if v > 0.5 else "#ef4444" for v in gvals]
bars = axes[1].barh(gnames, gvals, color=gcolors, edgecolor="white")
axes[1].axvline(0.5, color="red", linestyle="--", linewidth=1)
axes[1].set_xlabel("AUROC")
axes[1].set_title("Attention Entropy by Layer Group", fontweight="bold")
for bar, val in zip(bars, gvals):
    axes[1].text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=9)

plt.suptitle("Attention Entropy Revisited: Does Layer Choice Matter?",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "attention_by_layer.png"), bbox_inches="tight", dpi=150)
print(f"\nSaved attention_by_layer.png", flush=True)

# KEY FINDING
best_attn = max(per_layer_auroc)
print(f"\n{'='*60}", flush=True)
print("  CONCLUSION", flush=True)
print(f"{'='*60}", flush=True)
if best_attn > 0.6:
    print(f"  Earlier layers DO improve attention entropy!", flush=True)
    print(f"  Best layer {best_layer}: AUROC={best_attn:.4f} vs last layer: {per_layer_auroc[-1]:.4f}", flush=True)
    print(f"  However, still weaker than sampling-based methods (0.77-0.79)", flush=True)
else:
    print(f"  Attention entropy remains uninformative across ALL layers.", flush=True)
    print(f"  Best: {best_attn:.4f}, confirming this is not a layer-choice issue", flush=True)
    print(f"  but a fundamental limitation of attention patterns for QA hallucination.", flush=True)
print(f"{'='*60}", flush=True)

with open(os.path.join(SAVE_DIR, "attention_results.pkl"), "wb") as f:
    pickle.dump({"per_layer_auroc": per_layer_auroc, "group_aurocs": group_aurocs,
                 "best_layer": best_layer, "n_layers": n_layers}, f)
print("Saved attention_results.pkl", flush=True)

del model, tokenizer
torch.cuda.empty_cache()
print("\nDone!", flush=True)
input("Press Enter to exit...")