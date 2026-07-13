import sys
import os
import re
import time
import string
import pickle
import random
import traceback
from collections import Counter

print("Starting multi-dataset evaluation...", flush=True)
import numpy as np
import torch
from tqdm import tqdm

# ── SELECT DATASET ──
# python run_multidataset.py triviaqa
# python run_multidataset.py naturalqa
# python run_multidataset.py hotpotqa

DATASET_CONFIGS = {
    "triviaqa": {
        "hf_name": "mandarjoshi/trivia_qa",
        "hf_config": "rc.nocontext",
        "split": "validation",
        "display": "TriviaQA",
    },
    "naturalqa": {
        "hf_name": "google-research-datasets/nq_open",
        "hf_config": None,
        "split": "validation",
        "display": "Natural Questions (Open)",
    },
    "hotpotqa": {
        "hf_name": "hotpotqa/hotpot_qa",
        "hf_config": "fullwiki",
        "split": "validation",
        "display": "HotpotQA (Multi-hop)",
    },
    "medqa": {
        "hf_name": "GBaker/MedQA-USMLE-4-options",
        "hf_config": None,
        "split": "test",
        "display": "MedQA (Medical)",
    },
}

if len(sys.argv) < 2 or sys.argv[1] not in DATASET_CONFIGS:
    print("Usage: python run_multidataset.py [triviaqa|naturalqa|hotpotqa]", flush=True)
    print("Available datasets:", list(DATASET_CONFIGS.keys()), flush=True)
    sys.exit(1)

ds_key = sys.argv[1]
ds_config = DATASET_CONFIGS[ds_key]

SAVE_DIR = os.path.join(os.getcwd(), "results_" + ds_key)
os.makedirs(SAVE_DIR, exist_ok=True)
N_QUESTIONS = 200
N_SAMPLES = 10
TEMPERATURE = 0.7
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

print(f"Dataset: {ds_config['display']}", flush=True)
print(f"Model: {MODEL_NAME}", flush=True)
print(f"Results: {SAVE_DIR}", flush=True)

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

# ── LOAD MODEL ──
try:
    print("\nLoading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=qc, device_map="auto", dtype=torch.float16)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    print("Model loaded!", flush=True)
except Exception as e:
    print(f"MODEL FAILED: {e}", flush=True)
    traceback.print_exc()
    input("Press Enter..."); sys.exit(1)

def format_prompt(question):
    msgs = [{"role": "user", "content": "Answer the following question with a short, direct answer. Do not explain.\n\nQuestion: " + question + "\nAnswer:"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def generate(prompt, temp=0.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inlen = inputs["input_ids"].shape[1]
    kw = dict(**inputs, max_new_tokens=50, return_dict_in_generate=True, output_scores=True)
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

# ── LOAD DATASET ──
try:
    print(f"\nLoading {ds_config['display']}...", flush=True)
    from datasets import load_dataset

    if ds_config["hf_config"]:
        raw_ds = load_dataset(ds_config["hf_name"], ds_config["hf_config"], split=ds_config["split"])
    else:
        raw_ds = load_dataset(ds_config["hf_name"], split=ds_config["split"])

    # Extract questions and answers based on dataset format
    qa_pairs = []

    if ds_key == "triviaqa":
        for item in raw_ds:
            qa_pairs.append({
                "question": item["question"],
                "answers": item["answer"]["aliases"],
            })

    elif ds_key == "naturalqa":
        for item in raw_ds:
            qa_pairs.append({
                "question": item["question"],
                "answers": item["answer"],  # nq_open has list of answers
            })

    elif ds_key == "hotpotqa":
        for item in raw_ds:
            qa_pairs.append({
                "question": item["question"],
                "answers": [item["answer"]],  # hotpotqa has single answer string
            })

    # Sample 200
    random.seed(42)
    if len(qa_pairs) > N_QUESTIONS:
        qa_pairs = random.sample(qa_pairs, N_QUESTIONS)

    print(f"Loaded {len(qa_pairs)} questions", flush=True)
    print(f"Example: \"{qa_pairs[0]['question']}\"", flush=True)
    print(f"Answers: {qa_pairs[0]['answers'][:3]}", flush=True)

except Exception as e:
    print(f"DATASET FAILED: {e}", flush=True)
    traceback.print_exc()
    input("Press Enter..."); sys.exit(1)

# ── STEP 1: GREEDY ──
results = load_ckpt("step1")
if results and len(results) >= len(qa_pairs):
    print(f"\nStep 1 loaded ({len(results)} questions)", flush=True)
else:
    try:
        print(f"\n=== STEP 1: GREEDY BASELINE ({ds_config['display']}) ===", flush=True)
        results = []
        for i in tqdm(range(len(qa_pairs)), desc="Step 1"):
            q = qa_pairs[i]["question"]
            gt = qa_pairs[i]["answers"]
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
                print(f"  [{i+1}/{len(qa_pairs)}] acc: {acc:.1%}", flush=True)
        save_ckpt(results, "step1")
    except Exception as e:
        print(f"STEP 1 FAILED: {e}", flush=True)
        traceback.print_exc()
        if results: save_ckpt(results, "step1_partial")
        input("Press Enter..."); sys.exit(1)

greedy_acc = sum(r["is_correct"] for r in results) / len(results)
print(f"Greedy accuracy: {greedy_acc:.2%}", flush=True)

# ── STEP 2: SAMPLING ──
has_samples = all("samples" in r for r in results)
if has_samples:
    print("Step 2 loaded", flush=True)
else:
    try:
        print(f"\n=== STEP 2: SAMPLING ===", flush=True)
        start = 0
        ckpt2 = load_ckpt("step2")
        if ckpt2:
            start = len(ckpt2)
            for ii in range(min(start, len(results))):
                results[ii]["samples"] = ckpt2[ii]["samples"]
            print(f"  Resuming from {start}", flush=True)
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
        print(f"STEP 2 FAILED: {e}", flush=True)
        traceback.print_exc()
        input("Press Enter..."); sys.exit(1)

# Self-consistency
for r in results:
    ans = [normalize_answer(s["response"]) for s in r["samples"]]
    cts = Counter(ans)
    maj, majc = cts.most_common(1)[0]
    r["majority_answer"] = maj
    r["uncertainty_sc"] = 1.0 - majc / N_SAMPLES
    r["majority_correct"] = check_answer(maj, r["ground_truths"])

maj_acc = sum(r["majority_correct"] for r in results) / len(results)
print(f"Majority-vote: {maj_acc:.2%}", flush=True)

# ── STEP 3: SEMANTIC ENTROPY ──
if all("uncertainty_se" in r for r in results):
    print("Step 3 loaded", flush=True)
else:
    try:
        print(f"\n=== STEP 3: SEMANTIC ENTROPY ===", flush=True)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer as ST
        NLI = "microsoft/deberta-large-mnli"
        ntok = ST.from_pretrained(NLI)
        nmod = AutoModelForSequenceClassification.from_pretrained(NLI).eval()
        try:
            nmod = nmod.to("cuda"); ndev = "cuda"
        except:
            ndev = "cpu"
        def entails(a, b):
            inp = ntok(a, b, return_tensors="pt", truncation=True, max_length=256).to(ndev)
            with torch.no_grad(): return nmod(**inp).logits.argmax(-1).item() == 2
        def bidir(a, b): return entails(a, b) and entails(b, a)
        def do_cluster(responses):
            cls = []
            for rr in responses:
                if not rr.strip(): continue
                placed = False
                for c in cls:
                    if bidir(c[0], rr): c.append(rr); placed = True; break
                if not placed: cls.append([rr])
            return cls
        for i in tqdm(range(len(results)), desc="Step 3"):
            resps = [s["response"] for s in results[i]["samples"]]
            cl = do_cluster(resps)
            results[i]["n_semantic_clusters"] = len(cl)
            ps = np.array([len(c)/N_SAMPLES for c in cl])
            results[i]["uncertainty_se"] = float(-np.sum(ps * np.log(ps + 1e-12)))
            if (i+1) % 25 == 0: save_ckpt(results, "step3")
        save_ckpt(results, "step3")
        del nmod, ntok; torch.cuda.empty_cache()
    except Exception as e:
        print(f"STEP 3 FAILED: {e}", flush=True)
        traceback.print_exc()
        input("Press Enter..."); sys.exit(1)

# ── STEP 5: SELFCHECKGPT-NLI ──
if all("uncertainty_selfcheck" in r for r in results):
    print("Step 5 loaded", flush=True)
else:
    try:
        print(f"\n=== STEP 5: SELFCHECKGPT-NLI ===", flush=True)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer as ST
        NLI = "microsoft/deberta-large-mnli"
        ntok = ST.from_pretrained(NLI)
        nmod = AutoModelForSequenceClassification.from_pretrained(NLI).eval()
        try:
            nmod = nmod.to("cuda"); ndev = "cuda"
        except:
            ndev = "cpu"
        for i in tqdm(range(len(results)), desc="Step 5"):
            main = results[i]["greedy_response"]
            samps = [s["response"] for s in results[i]["samples"]]
            sc_list = []
            for sa in samps:
                if not sa.strip() or not main.strip():
                    sc_list.append(0.5); continue
                prem = "The answer to the question is: " + sa
                hyp = "The answer to the question is: " + main
                inp = ntok(prem, hyp, return_tensors="pt", truncation=True, max_length=256).to(ndev)
                with torch.no_grad(): logits = nmod(**inp).logits
                ze, zc = logits[0, 2].item(), logits[0, 0].item()
                sc_list.append(np.exp(zc) / (np.exp(ze) + np.exp(zc)))
            results[i]["uncertainty_selfcheck"] = float(np.mean(sc_list))
            if (i+1) % 25 == 0: save_ckpt(results, "step5")
        save_ckpt(results, "step5")
        del nmod, ntok; torch.cuda.empty_cache()
    except Exception as e:
        print(f"STEP 5 FAILED: {e}", flush=True)
        traceback.print_exc()
        input("Press Enter..."); sys.exit(1)

# ── EVALUATION ──
try:
    print(f"\n=== EVALUATION ({ds_config['display']}) ===", flush=True)
    from sklearn.metrics import roc_auc_score, average_precision_score

    labels = np.array([float(not r["is_correct"]) for r in results])

    # Check we have both classes
    if labels.sum() == 0 or labels.sum() == len(labels):
        print("WARNING: All answers are the same class. AUROC undefined.", flush=True)
        print(f"  Correct: {int((1-labels).sum())}  Wrong: {int(labels.sum())}", flush=True)
    else:
        def mnorm(x):
            rng = x.max() - x.min()
            return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)

        sc_scores = {
            "Avg(-logP)": np.array([r["avg_neg_log_prob"] for r in results]),
            "Self-Consistency": np.array([r["uncertainty_sc"] for r in results]),
            "Semantic Entropy": np.array([r["uncertainty_se"] for r in results]),
            "SelfCheckGPT-NLI": np.array([r["uncertainty_selfcheck"] for r in results]),
        }
        sc_scores["Combined"] = (
            mnorm(sc_scores["Self-Consistency"]) + mnorm(sc_scores["Semantic Entropy"])
            + mnorm(sc_scores["Avg(-logP)"])
        ) / 3.0

        print(f"\n{'Method':<28} {'AUROC':<9} {'AUPRC':<9}", flush=True)
        print("-" * 46, flush=True)
        ev = {}
        for name, sc in sc_scores.items():
            auroc = roc_auc_score(labels, sc) if np.std(sc) > 1e-10 else 0.5
            auprc = average_precision_score(labels, sc) if np.std(sc) > 1e-10 else float(labels.mean())
            ev[name] = {"auroc": auroc, "auprc": auprc}
            print(f"{name:<28} {auroc:<9.4f} {auprc:<9.4f}", flush=True)

        # Fingerprinting
        sc_vals = sc_scores["Self-Consistency"]
        se_vals = sc_scores["Semantic Entropy"]
        sc_med = np.median(sc_vals)
        se_med = np.median(se_vals)

        for r in results:
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

        type_counts = Counter(r["fingerprint_type"] for r in results)

        print(f"\nFingerprint distribution:", flush=True)
        for ft, c in type_counts.most_common():
            hall = sum(1 for r in results if r["fingerprint_type"] == ft and not r["is_correct"])
            rate = hall / c if c > 0 else 0
            print(f"  {ft:<25} n={c:<4} hall_rate={rate:.0%}", flush=True)

        # Adaptive policy
        answered = 0
        correct_ans = 0
        abstained = 0
        for r in results:
            ft = r["fingerprint_type"]
            if ft in ("knowledge_gap",):
                abstained += 1
            else:
                answered += 1
                if r["is_correct"]:
                    correct_ans += 1

        policy_acc = correct_ans / answered if answered > 0 else 0
        coverage = answered / len(results)
        print(f"\nAdaptive policy:", flush=True)
        print(f"  Baseline accuracy:    {greedy_acc:.2%}", flush=True)
        print(f"  Policy accuracy:      {policy_acc:.2%}", flush=True)
        print(f"  Coverage:             {coverage:.1%}", flush=True)
        print(f"  Improvement:          {policy_acc - greedy_acc:+.2%}", flush=True)

        save_ckpt({
            "results": results, "eval": ev,
            "model": MODEL_NAME, "dataset": ds_config["display"],
            "greedy_acc": greedy_acc, "majority_acc": maj_acc,
            "type_counts": dict(type_counts),
            "policy_acc": policy_acc, "coverage": coverage,
        }, "final_results")

    print(f"\n{'='*60}", flush=True)
    print(f"  {ds_config['display']} COMPLETE", flush=True)
    print(f"  Greedy: {greedy_acc:.2%}  Majority: {maj_acc:.2%}", flush=True)
    print(f"{'='*60}", flush=True)

except Exception as e:
    print(f"EVAL FAILED: {e}", flush=True)
    traceback.print_exc()
    save_ckpt(results, "results_backup")

print("\nDone!", flush=True)
input("Press Enter to exit...")
