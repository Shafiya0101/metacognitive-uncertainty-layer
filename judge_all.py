"""
judge_all.py - label EVERY saved answer with the Claude judge (resumable).

Run from the project folder, with the key set in this PowerShell window:
    $env:ANTHROPIC_API_KEY = "paste-your-key-here"
    .\\venv\\Scripts\\python.exe judge_all.py

Writes analysis_out/judge_labels.json. If it stops (network, limits), just run
it again: finished answers are kept and only the missing ones are re-judged.
About 1,000 short calls.
"""
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

SOURCES = {
    "Mistral-7B | TriviaQA": os.path.join("results", "final_results.pkl"),
    "Qwen2-7B | TriviaQA": os.path.join("results_llama3", "final_results.pkl"),
    "Qwen2-1.5B | TriviaQA": os.path.join("results_phi3", "final_results.pkl"),
    "Mistral-7B | NaturalQA": os.path.join("results_naturalqa", "final_results.pkl"),
    "Mistral-7B | HotpotQA": os.path.join("results_hotpotqa", "final_results.pkl"),
}
OUT = "analysis_out"
CACHE = os.path.join(OUT, "judge_labels.json")
os.makedirs(OUT, exist_ok=True)

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY is not set in this PowerShell window.")
import anthropic
client = anthropic.Anthropic(max_retries=5)


def judge(q, gold, ans):
    prompt = (f"Question: {q}\nAccepted answers: {gold}\nModel answer: {ans}\n\n"
              "Does the model answer state the same fact as one of the accepted "
              "answers? Reply with exactly one word: CORRECT, INCORRECT, "
              "or AMBIGUOUS.")
    kw = dict(model="claude-haiku-5-5", max_tokens=3000,
              messages=[{"role": "user", "content": prompt}])
    try:
        r = client.messages.create(temperature=0, **kw)
    except Exception:
        r = client.messages.create(**kw)
    text = "".join(b.text for b in r.content
                   if getattr(b, "type", "") == "text").strip().upper()
    if text.startswith("CORRECT"):
        return "CORRECT"
    if text.startswith("INCORRECT"):
        return "INCORRECT"
    if "AMBIG" in text:
        return "AMBIGUOUS"
    return "OTHER"


cache = {}
if os.path.exists(CACHE):
    with open(CACHE, encoding="utf-8") as f:
        cache = json.load(f)

tasks = []
for name, path in SOURCES.items():
    if not os.path.exists(path):
        print("MISSING FILE:", path)
        continue
    with open(path, "rb") as f:
        d = pickle.load(f)
    res = d["results"] if isinstance(d, dict) else d
    for r in res:
        key = f"{name}#{r['idx']}"
        if key not in cache:
            gold = " | ".join(str(g) for g in r["ground_truths"][:8])
            tasks.append((key, r["question"], gold, r["greedy_response"]))

print(f"{len(cache)} already judged, {len(tasks)} to do")


def save():
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f)


with ThreadPoolExecutor(max_workers=6) as ex:
    futs = {ex.submit(judge, q, g, a): key for key, q, g, a in tasks}
    for n, fut in enumerate(as_completed(futs), 1):
        key = futs[fut]
        try:
            cache[key] = fut.result()
        except Exception as e:
            print("error on", key, "->", str(e)[:120])
            continue
        if n % 25 == 0:
            save()
            print(f"  {n}/{len(tasks)} done")
save()

counts = {}
for v in cache.values():
    counts[v] = counts.get(v, 0) + 1
print("\nVerdict counts:", counts)
print("Saved:", CACHE)
