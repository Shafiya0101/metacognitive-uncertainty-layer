"""
show_disagreements.py - list only the rows where the Claude judge disagrees
with the original (containment-matching) label, so you can review them by hand.

Run from the project folder:
    .\\venv\\Scripts\\python.exe show_disagreements.py
Reads  analysis_out/label_check.csv  ->  writes analysis_out/disagreements.csv
"""
import csv
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SRC = os.path.join("analysis_out", "label_check.csv")
DST = os.path.join("analysis_out", "disagreements.csv")

with open(SRC, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

dis = []
for r in rows:
    judge_correct = r["claude_judge"].startswith("CORRECT")
    orig_correct = r["original_label"] == "correct"
    if judge_correct != orig_correct:
        r["direction"] = ("original=correct, judge=NOT correct" if orig_correct
                          else "original=wrong, judge=CORRECT")
        dis.append(r)

lines = [f"{len(dis)} disagreements out of {len(rows)} rows\n"]
for n, d in enumerate(dis, 1):
    lines.append("-" * 70)
    lines.append(f"CASE {n}: {d['source']} | idx {d['idx']} | {d['group']} | {d['direction']}")
    lines.append(f"  Q:     {d['question'][:200]}")
    lines.append(f"  Gold:  {d['gold_answers'][:200]}")
    lines.append(f"  Model: {d['model_answer'][:300]}")
    lines.append(f"  Judge said: {d['claude_judge']}")
lines.append("\n" + "=" * 70)
for k in ("original=wrong, judge=CORRECT", "original=correct, judge=NOT correct"):
    lines.append(f"{k}: {sum(1 for d in dis if d['direction'] == k)}")

TXT = os.path.join("analysis_out", "disagreements.txt")
with open(TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
print("\nAlso saved as a text file:", TXT)

if dis:
    with open(DST, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(dis[0].keys()))
        w.writeheader()
        w.writerows(dis)
    print("\nSaved:", DST)
