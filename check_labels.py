"""
check_labels.py - export answers for manual label checking (+ optional Claude judge).

Run from the project folder:
    .\\venv\\Scripts\\python.exe check_labels.py            (export only, no API)
    .\\venv\\Scripts\\python.exe check_labels.py --judge    (also ask Claude)

For each of the 5 sources it exports
  * every "strict silent error": model wrong, but all 10 samples agreed
  * 40 random questions (to estimate how often the original label is wrong)
to  analysis_out/label_check.csv  (open it in Excel).

The API key is read from the environment variable ANTHROPIC_API_KEY.
Never write the key into this file.
"""
import csv
import os
import pickle
import random
import sys
import numpy as np

SOURCES = {
    "Mistral-7B | TriviaQA": os.path.join("results", "final_results.pkl"),
    "Qwen2-7B | TriviaQA": os.path.join("results_llama3", "final_results.pkl"),
    "Qwen2-1.5B | TriviaQA": os.path.join("results_phi3", "final_results.pkl"),
    "Mistral-7B | NaturalQA": os.path.join("results_naturalqa", "final_results.pkl"),
    "Mistral-7B | HotpotQA": os.path.join("results_hotpotqa", "final_results.pkl"),
}
N_RANDOM = 40
USE_JUDGE = "--judge" in sys.argv
OUT = "analysis_out"
os.makedirs(OUT, exist_ok=True)
random.seed(42)

client = None
if USE_JUDGE:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set in this PowerShell window.")
    import anthropic
    client = anthropic.Anthropic()


def judge(q, gold, ans):
    prompt = (f"Question: {q}\nAccepted answers: {gold}\nModel answer: {ans}\n\n"
              "Does the model answer state the same fact as one of the accepted "
              "answers? Reply with exactly one word: CORRECT, INCORRECT, "
              "or AMBIGUOUS.")
    # max_tokens is generous because the model may "think" before answering
    kw = dict(model="claude-haiku-5-5", max_tokens=3000,
              messages=[{"role": "user", "content": prompt}])
    try:
        r = client.messages.create(temperature=0, **kw)
    except Exception:
        r = client.messages.create(**kw)
    # The reply can start with a thinking block: keep only the text block(s)
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return text.strip().upper() or "NO_TEXT"


rows = []
for name, path in SOURCES.items():
    if not os.path.exists(path):
        print("MISSING FILE:", path)
        continue
    with open(path, "rb") as f:
        d = pickle.load(f)
    res = d["results"] if isinstance(d, dict) else d
    silent, others = [], []
    for i, r in enumerate(res):
        unanimous = (r["uncertainty_sc"] <= 1e-9
                     and abs(r["uncertainty_se"]) <= 1e-6)
        (silent if (unanimous and not r["is_correct"]) else others).append(i)
    pick = [(i, "silent_error") for i in silent]
    pick += [(i, "random") for i in random.sample(others, min(N_RANDOM, len(others)))]
    print(f"{name}: {len(silent)} silent errors + "
          f"{len(pick) - len(silent)} random")
    for i, group in pick:
        r = res[i]
        gold = " | ".join(str(g) for g in r["ground_truths"][:8])
        verdict = judge(r["question"], gold, r["greedy_response"]) if USE_JUDGE else ""
        rows.append({
            "source": name, "idx": r["idx"], "group": group,
            "question": r["question"][:300], "gold_answers": gold,
            "model_answer": r["greedy_response"],
            "original_label": "correct" if r["is_correct"] else "wrong",
            "claude_judge": verdict, "my_verdict(correct/wrong/ambiguous)": "",
        })

path = os.path.join(OUT, "label_check.csv")
with open(path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {len(rows)} rows to {path}")

if USE_JUDGE:
    dis = [r for r in rows
           if (r["claude_judge"].startswith("CORRECT")) != (r["original_label"] == "correct")]
    print(f"Claude disagrees with the original label on {len(dis)} of {len(rows)} rows")
    for g in ("silent_error", "random"):
        sub = [r for r in rows if r["group"] == g]
        dd = [r for r in dis if r["group"] == g]
        print(f"  {g}: {len(dd)} of {len(sub)}")
