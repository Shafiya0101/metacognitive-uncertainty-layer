import sys
import os
import re
import time
import string
import pickle
import random
import traceback
from collections import Counter

print("Starting imports...", flush=True)
import numpy as np
import torch
from tqdm import tqdm
print("Imports OK", flush=True)

SAVE_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(SAVE_DIR, exist_ok=True)
N_QUESTIONS = 200
N_SAMPLES = 10
TEMPERATURE = 0.7

def save_ckpt(data, name):
    with open(os.path.join(SAVE_DIR, name + ".pkl"), "wb") as f:
        pickle.dump(data, f)
    print(f"  Saved {name}.pkl", flush=True)

def load_ckpt(name):
    p = os.path.join(SAVE_DIR, name + ".pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def check_answer(pred, gts):
    p = normalize_answer(pred)
    if not p:
        return False
    for g in gts:
        gn = normalize_answer(g)
        if gn and (gn in p or p in gn):
            return True
    return False

try:
    print("", flush=True)
    print("=" * 60, flush=True)
    print("  LOADING MODEL", flush=True)
    print("=" * 60, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

    qc = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("GPU: " + torch.cuda.get_device_name(0), flush=True)
    print("Loading " + MODEL_NAME + "...", flush=True)
    sys.stdout.flush()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizer OK", flush=True)
    sys.stdout.flush()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=qc,
        device_map="auto",
        dtype=torch.float16,
    )
    print("Model OK", flush=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

except Exception as e:
    print("MODEL LOADING FAILED: " + str(e), flush=True)
    traceback.print_exc()
    input("Press Enter to exit...")
    sys.exit(1)

def format_prompt(question):
    msgs = [{"role": "user", "content": "Answer the following question with a short, direct answer. Do not explain.\n\nQuestion: " + question + "\nAnswer:"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def generate(prompt, temp=0.0, max_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inlen = inputs["input_ids"].shape[1]
    kw = dict(**inputs, max_new_tokens=max_tokens, return_dict_in_generate=True, output_scores=True)
    if temp > 0:
        kw.update(do_sample=True, temperature=temp, top_p=0.95)
    else:
        kw.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**kw)
    gids = out.sequences[0][inlen:]
    text = tokenizer.decode(gids, skip_special_tokens=True).strip()
    lps = []
    for i, sc in enumerate(out.scores):
        if i < len(gids):
            pr = torch.softmax(sc[0], dim=-1)
            lps.append(torch.log(pr[gids[i]] + 1e-12).item())
    return text, lps

try:
    print("\nLoading TriviaQA...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    random.seed(42)
    idx = random.sample(range(len(ds)), N_QUESTIONS)
    subset = ds.select(idx)
    print("Loaded " + str(len(subset)) + " questions", flush=True)
except Exception as e:
    print("DATASET FAILED: " + str(e), flush=True)
    traceback.print_exc()
    input("Press Enter to exit...")
    sys.exit(1)

results = load_ckpt("step1")
if results and len(results) >= N_QUESTIONS:
    print("\nStep 1 loaded from checkpoint", flush=True)
else:
    try:
        print("\n" + "=" * 60, flush=True)
        print("  STEP 1: GREEDY BASELINE", flush=True)
        print("=" * 60, flush=True)
        results = []
        t0 = time.time()
        for i in tqdm(range(len(subset)), desc="Step 1"):
            q = subset[i]["question"]
            gt = subset[i]["answer"]["aliases"]
            resp, lps = generate(format_prompt(q), temp=0.0)
            correct = check_answer(resp, gt)
            results.append({
                "idx": i, "question": q, "ground_truths": gt,
                "greedy_response": resp, "is_correct": correct,
                "avg_neg_log_prob": -np.mean(lps) if lps else 999.0,
                "max_neg_log_prob": -min(lps) if lps else 999.0,
                "log_probs": lps,
            })
            if (i+1) % 50 == 0:
                acc = sum(r["is_correct"] for r in results) / len(results)
                print("  [" + str(i+1) + "/200] acc: " + f"{acc:.1%}", flush=True)
        save_ckpt(results, "step1")
    except Exception as e:
        print("STEP 1 FAILED: " + str(e), flush=True)
        traceback.print_exc()
        if results:
            save_ckpt(results, "step1_partial")
        input("Press Enter to exit...")
        sys.exit(1)

greedy_acc = sum(r["is_correct"] for r in results) / len(results)
print("Greedy accuracy: " + f"{greedy_acc:.2%}", flush=True)

has_samples = all("samples" in r for r in results)
if has_samples:
    print("\nStep 2 loaded", flush=True)
else:
    try:
        print("\n" + "=" * 60, flush=True)
        print("  STEP 2: SAMPLING (10 per question)", flush=True)
        print("=" * 60, flush=True)
        start = 0
        ckpt2 = load_ckpt("step2")
        if ckpt2:
            start = len(ckpt2)
            for ii in range(min(start, len(results))):
                results[ii]["samples"] = ckpt2[ii]["samples"]
            print("  Resuming from question " + str(start), flush=True)
        t0 = time.time()
        for i in tqdm(range(start, len(results)), desc="Step 2"):
            prompt = format_prompt(results[i]["question"])
            samps = []
            for _ in range(N_SAMPLES):
                r2, lp2 = generate(prompt, temp=TEMPERATURE)
                samps.append({"response": r2, "log_probs": lp2})
            results[i]["samples"] = samps
            if (i+1) % 10 == 0:
                c2 = [{"idx": rr["idx"], "samples": rr["samples"]} for rr in results[:i+1] if "samples" in rr]
                save_ckpt(c2, "step2")
        c2 = [{"idx": rr["idx"], "samples": rr["samples"]} for rr in results if "samples" in rr]
        save_ckpt(c2, "step2")
    except Exception as e:
        print("STEP 2 FAILED: " + str(e), flush=True)
        traceback.print_exc()
        c2 = [{"idx": rr["idx"], "samples": rr["samples"]} for rr in results if "samples" in rr]
        if c2:
            save_ckpt(c2, "step2")
        input("Press Enter to exit...")
        sys.exit(1)

for r in results:
    ans = [normalize_answer(s["response"]) for s in r["samples"]]
    cts = Counter(ans)
    maj, majc = cts.most_common(1)[0]
    r["majority_answer"] = maj
    r["consistency_score"] = majc / N_SAMPLES
    r["uncertainty_sc"] = 1.0 - r["consistency_score"]
    r["majority_correct"] = check_answer(maj, r["ground_truths"])

maj_acc = sum(r["majority_correct"] for r in results) / len(results)
print("Greedy: " + f"{greedy_acc:.2%}" + "  Majority-vote: " + f"{maj_acc:.2%}", flush=True)

if all("uncertainty_se" in r for r in results):
    print("\nStep 3 loaded", flush=True)
else:
    try:
        print("\n" + "=" * 60, flush=True)
        print("  STEP 3: SEMANTIC ENTROPY", flush=True)
        print("=" * 60, flush=True)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer as ST
        NLI = "microsoft/deberta-large-mnli"
        print("Loading " + NLI + "...", flush=True)
        ntok = ST.from_pretrained(NLI)
        nmod = AutoModelForSequenceClassification.from_pretrained(NLI).eval()
        try:
            nmod = nmod.to("cuda")
            ndev = "cuda"
        except:
            ndev = "cpu"
        print("  DeBERTa on " + ndev, flush=True)
        def entails(a, b):
            inp = ntok(a, b, return_tensors="pt", truncation=True, max_length=256).to(ndev)
            with torch.no_grad():
                return nmod(**inp).logits.argmax(-1).item() == 2
        def bidir_entail(a, b):
            return entails(a, b) and entails(b, a)
        def do_cluster(responses):
            cls = []
            for rr in responses:
                if not rr.strip():
                    continue
                placed = False
                for c in cls:
                    if bidir_entail(c[0], rr):
                        c.append(rr)
                        placed = True
                        break
                if not placed:
                    cls.append([rr])
            return cls
        def sem_entropy(cls, n):
            if not cls:
                return 0.0
            ps = np.array([len(c)/n for c in cls])
            return float(-np.sum(ps * np.log(ps + 1e-12)))
        t0 = time.time()
        for i in tqdm(range(len(results)), desc="Step 3"):
            resps = [s["response"] for s in results[i]["samples"]]
            cl = do_cluster(resps)
            results[i]["n_semantic_clusters"] = len(cl)
            results[i]["uncertainty_se"] = sem_entropy(cl, N_SAMPLES)
            if (i+1) % 25 == 0:
                save_ckpt(results, "step3")
        save_ckpt(results, "step3")
        del nmod, ntok
        torch.cuda.empty_cache()
    except Exception as e:
        print("STEP 3 FAILED: " + str(e), flush=True)
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)

if all("uncertainty_attn" in r for r in results):
    print("\nStep 3b loaded", flush=True)
else:
    try:
        print("\n" + "=" * 60, flush=True)
        print("  STEP 3b: ATTENTION ENTROPY", flush=True)
        print("=" * 60, flush=True)
        t0 = time.time()
        for i in tqdm(range(len(results)), desc="Step 3b"):
            prompt = format_prompt(results[i]["question"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50, do_sample=False, output_attentions=True, return_dict_in_generate=True)
            ents = []
            if hasattr(out, "attentions") and out.attentions:
                for sa in out.attentions:
                    if sa and len(sa) > 0:
                        aw = sa[-1][0, :, 0, :].float()
                        h = -(aw * torch.log(aw + 1e-12)).sum(-1).mean().item()
                        ents.append(h)
            results[i]["uncertainty_attn"] = float(np.mean(ents)) if ents else 0.0
            if (i+1) % 50 == 0:
                save_ckpt(results, "step3b")
        save_ckpt(results, "step3b")
    except Exception as e:
        print("STEP 3b FAILED: " + str(e), flush=True)
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)

if all("uncertainty_selfcheck" in r for r in results):
    print("\nStep 5 loaded", flush=True)
else:
    try:
        print("\n" + "=" * 60, flush=True)
        print("  STEP 5: SELFCHECKGPT-NLI", flush=True)
        print("=" * 60, flush=True)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer as ST
        NLI = "microsoft/deberta-large-mnli"
        print("Loading " + NLI + "...", flush=True)
        ntok = ST.from_pretrained(NLI)
        nmod = AutoModelForSequenceClassification.from_pretrained(NLI).eval()
        try:
            nmod = nmod.to("cuda")
            ndev = "cuda"
        except:
            ndev = "cpu"
        t0 = time.time()
        for i in tqdm(range(len(results)), desc="Step 5"):
            main = results[i]["greedy_response"]
            samps = [s["response"] for s in results[i]["samples"]]
            sc_list = []
            for sa in samps:
                if not sa.strip() or not main.strip():
                    sc_list.append(0.5)
                    continue
                prem = "The answer to the question is: " + sa
                hyp = "The answer to the question is: " + main
                inp = ntok(prem, hyp, return_tensors="pt", truncation=True, max_length=256).to(ndev)
                with torch.no_grad():
                    logits = nmod(**inp).logits
                ze = logits[0, 2].item()
                zc = logits[0, 0].item()
                sc_list.append(np.exp(zc) / (np.exp(ze) + np.exp(zc)))
            results[i]["uncertainty_selfcheck"] = float(np.mean(sc_list))
            if (i+1) % 25 == 0:
                save_ckpt(results, "step5")
        save_ckpt(results, "step5")
        del nmod, ntok
        torch.cuda.empty_cache()
    except Exception as e:
        print("STEP 5 FAILED: " + str(e), flush=True)
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)

try:
    print("\n" + "=" * 60, flush=True)
    print("  EVALUATION", flush=True)
    print("=" * 60, flush=True)
    from sklearn.metrics import roc_auc_score, average_precision_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = np.array([float(not r["is_correct"]) for r in results])
    def mnorm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)

    sc_scores = {
        "Avg(-logP)": np.array([r["avg_neg_log_prob"] for r in results]),
        "Self-Consistency": np.array([r["uncertainty_sc"] for r in results]),
        "Semantic Entropy": np.array([r["uncertainty_se"] for r in results]),
        "Attention Entropy": np.array([r["uncertainty_attn"] for r in results]),
        "SelfCheckGPT-NLI": np.array([r["uncertainty_selfcheck"] for r in results]),
    }
    sc_scores["Combined (4 methods)"] = (
        mnorm(sc_scores["Self-Consistency"]) + mnorm(sc_scores["Semantic Entropy"])
        + mnorm(sc_scores["Attention Entropy"]) + mnorm(sc_scores["Avg(-logP)"])
    ) / 4.0

    print("", flush=True)
    print(f"{'Method':<28} {'AUROC':<9} {'AUPRC':<9}", flush=True)
    print("-" * 46, flush=True)
    ev = {}
    for name, sc in sc_scores.items():
        auroc = roc_auc_score(labels, sc) if np.std(sc) > 1e-10 else 0.5
        auprc = average_precision_score(labels, sc) if np.std(sc) > 1e-10 else float(labels.mean())
        ev[name] = {"auroc": auroc, "auprc": auprc}
        print(f"{name:<28} {auroc:<9.4f} {auprc:<9.4f}", flush=True)

    confs = np.clip(np.array([np.exp(-r["avg_neg_log_prob"]) for r in results]), 0, 1)
    accs_arr = np.array([float(r["is_correct"]) for r in results])
    bins_e = np.linspace(0, 1, 11)
    ece = 0.0
    ece_bins = []
    for b in range(10):
        m = (confs >= bins_e[b]) & (confs < bins_e[b+1])
        n = m.sum()
        if n > 0:
            ac = float(accs_arr[m].mean())
            cn = float(confs[m].mean())
            ece += (n/len(confs)) * abs(ac - cn)
            ece_bins.append({"conf": cn, "acc": ac, "n": int(n)})
    print(f"\nECE: {ece:.4f}", flush=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(ev.keys())
    aurocs = [ev[n]["auroc"] for n in names]
    colors = ["#888780", "#5DCAA5", "#AFA9EC", "#F4C0D1", "#F0997B", "#85B7EB"]
    ax.barh(names, aurocs, color=colors[:len(names)], edgecolor="white", height=0.6)
    ax.set_xlim(0.4, max(aurocs) + 0.06)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("AUROC")
    ax.set_title("Hallucination Detection: AUROC Comparison")
    for bar, val in zip(ax.patches, aurocs):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "auroc_comparison.png"), bbox_inches="tight")
    print("Saved auroc_comparison.png", flush=True)

    correct = np.array([float(r["is_correct"]) for r in results])
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, color in [("Self-Consistency", "#5DCAA5"), ("Semantic Entropy", "#AFA9EC"),
                         ("Attention Entropy", "#ED93B1"), ("SelfCheckGPT-NLI", "#F0997B"),
                         ("Combined (4 methods)", "#85B7EB")]:
        sidx = np.argsort(sc_scores[name])
        sc_sorted = correct[sidx]
        cov = np.arange(1, len(sc_sorted)+1) / len(sc_sorted)
        ac = np.cumsum(sc_sorted) / np.arange(1, len(sc_sorted)+1)
        aurac = float(np.trapezoid(ac, cov))
        ev[name]["aurac"] = aurac
        ax.plot(cov, ac, label=f"{name} ({aurac:.3f})", linewidth=2, color=color)
    bl = correct.mean()
    ax.axhline(bl, color="gray", linestyle="--", linewidth=0.8, label=f"No abstention: {bl:.1%}")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy-Coverage Tradeoff")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(0, 1.02)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "accuracy_coverage.png"), bbox_inches="tight")
    print("Saved accuracy_coverage.png", flush=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    if ece_bins:
        ax.bar([b["conf"] for b in ece_bins], [b["acc"] for b in ece_bins], width=0.08, alpha=0.7, color="#AFA9EC", edgecolor="white")
    ax.plot([0,1], [0,1], "k--", linewidth=0.8)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Calibration - ECE = {ece:.4f}")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "calibration_diagram.png"), bbox_inches="tight")
    print("Saved calibration_diagram.png", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cm = correct.astype(bool)
    for ax, (name, col) in zip(axes.flatten(),
        [("Self-Consistency","#5DCAA5"), ("Semantic Entropy","#AFA9EC"),
         ("Attention Entropy","#ED93B1"), ("SelfCheckGPT-NLI","#F0997B")]):
        sc = sc_scores[name]
        ax.hist(sc[cm], bins=20, alpha=0.6, color="#97C459", label="Correct", density=True)
        ax.hist(sc[~cm], bins=20, alpha=0.6, color="#E24B4A", label="Wrong", density=True)
        ax.set_title(f"{name} (AUROC={ev[name]['auroc']:.3f})")
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "uncertainty_distributions.png"), bbox_inches="tight")
    print("Saved uncertainty_distributions.png", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  FINAL RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"  Model: {MODEL_NAME}", flush=True)
    print(f"  Questions: {len(results)}", flush=True)
    print(f"  Greedy accuracy: {greedy_acc:.2%}", flush=True)
    print(f"  Majority-vote:   {maj_acc:.2%}", flush=True)
    print(f"  ECE: {ece:.4f}", flush=True)
    print("", flush=True)
    print(f"  {'Method':<28} {'AUROC':<9} {'AUPRC':<9}", flush=True)
    print("  " + "-" * 46, flush=True)
    for n in names:
        print(f"  {n:<28} {ev[n]['auroc']:<9.4f} {ev[n]['auprc']:<9.4f}", flush=True)
    comb = ev["Combined (4 methods)"]["auroc"]
    best = max(ev[m]["auroc"] for m in ["Self-Consistency","Semantic Entropy","Attention Entropy","SelfCheckGPT-NLI"])
    bestn = max(["Self-Consistency","Semantic Entropy","Attention Entropy","SelfCheckGPT-NLI"], key=lambda m: ev[m]["auroc"])
    print(f"\n  Combined: {comb:.4f} vs Best: {best:.4f} ({bestn})", flush=True)
    print(f"  Improvement: {comb-best:+.4f}", flush=True)
    print("=" * 60, flush=True)
    save_ckpt({"results": results, "eval": ev, "ece": ece, "model": MODEL_NAME}, "final_results")
except Exception as e:
    print("EVALUATION FAILED: " + str(e), flush=True)
    traceback.print_exc()
    save_ckpt(results, "results_backup")

print("\nALL DONE!", flush=True)
print("Results in: " + SAVE_DIR, flush=True)
input("\nPress Enter to exit...")
