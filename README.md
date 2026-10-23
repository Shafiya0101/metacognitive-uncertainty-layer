# Uncertainty Signals for LLM Hallucination Detection

Code, results and analysis scripts for an empirical study of whether
combining LLM uncertainty signals, or reading their joint pattern (a
"fingerprint"), improves hallucination detection and abstention in question
answering.

> **Update after the thesis.** This repository began as a final-year project
> (aivancity PGE5, 2025-2026). Later re-analysis with confidence intervals,
> label-free definitions and a label audit corrected several thesis-era claims.
> The section "Corrections to the thesis" lists them. The analysis scripts
> `analysis.py`, `analysis2.py` and `analysis3.py` produce the current numbers.

## What the study does

For each question, a base LLM gives a greedy answer and 10 sampled answers.
Four uncertainty signals are computed:

| Signal | Access | What it measures |
|---|---|---|
| Log-probability (LP) | grey-box | token confidence in the greedy answer |
| Self-consistency (SC) | black-box | string agreement among samples |
| Semantic entropy (SE) | black-box | agreement in meaning among samples (NLI clustering) |
| SelfCheckGPT-NLI | black-box | contradiction between samples and the greedy answer |

Settings (200 questions each): Mistral-7B-Instruct-v0.3, Qwen2-7B-Instruct and
Qwen2-1.5B-Instruct on TriviaQA; Mistral-7B on Natural Questions and HotpotQA.
All models run in 4-bit NF4 on one 8 GB laptop GPU.

## Main findings (containment labels, 95% bootstrap intervals)

* Detection AUROC of single signals is 0.63-0.82. SelfCheckGPT-NLI is the
  strongest single signal on all three TriviaQA runs.
* Combining signals does **not** significantly beat SelfCheckGPT-NLI. Paired
  differences all include zero, except the three-signal combination on
  Qwen2-1.5B, where it is significantly worse (-0.072).
* SC and SE are strongly rank-correlated (Spearman 0.86-0.93). A
  likelihood-ratio test finds no robust SC x SE interaction.
* A label-free fingerprint abstention policy is **not** better than a plain
  threshold at matched coverage: no interval is above zero in 60 comparisons,
  nor in 100 comparisons across the 30th-70th percentile cut-offs.
* Defining error types with correctness labels inflates results: 93.9% accuracy
  at 81.5% coverage (label-dependent) vs. 91.2% at 56.5% coverage (label-free).
* Unanimous agreement among 10 samples is not a guarantee: 5 / 12 / 0 / 4 / 5
  wrong answers in the five settings (4 / 16 / 0 / 4 / 4 with LLM-judge labels).
* Label audit: of 200 random answers, 11 (5.5%) carried a wrong containment
  label (7.5% counting ambiguous cases).

Full numbers: `analysis_out/summary.txt`, `summary_absolute.txt`,
`summary_threshold.txt`, and the same files in `analysis_out_judge/` for the
LLM-judge labels.

## Corrections to the thesis

| Thesis / earlier README claim | Status |
|---|---|
| Fingerprint policy: 93.87% accuracy at 81.5% coverage (+17.37 pp) | Used correctness labels to define types. Label-free: 91.2% at 56.5% coverage, and not better than thresholding. |
| Combined method outperforms SelfCheckGPT-NLI | Not statistically significant. |
| Smaller models produce 6x more confident fabrications | Artifact of dataset-relative median thresholds. With "all 10 samples identical", counts are 5, 12 and 0 per 200. |
| MedQA: model wrong 91.5%, "proves uncertainty estimation is insufficient for medical AI" | Invalid. MedQA was evaluated as open-ended generation without answer options, so 8.5% is not comparable to multiple-choice chance. Do not interpret. |
| Learned classifier improves every source by up to +0.06 | Computed in-sample. Cross-validated, it is worse than equal weighting in 4 of 5 settings. |
| Attention entropy: "NF4 destroys attention information" | Attention extraction returned zeros in this setup; the cause was not investigated and the signal is not evaluated. |
| Fingerprint-enhanced beats the learned meta-classifier | Not supported. |

## Reproduce

```
pip install -r requirements.txt
python analysis.py                 # AUROC, regions, policy vs thresholds, figures
python analysis2.py                # strict unanimous errors, SC-SE correlation
python analysis3.py                # threshold robustness (percentile cut-offs)
python analysis.py --judge-labels  # same with LLM-judge labels
```

The analyses read the saved per-question results (`results*/final_results.pkl`)
and need no GPU. These are Python pickles: only load files you trust.

Label audit (needs an Anthropic API key in the `ANTHROPIC_API_KEY` environment
variable; no key is stored in this repository):

```
python judge_all.py                # LLM-judge label for every answer
python check_labels.py --judge     # export audit sample
python show_disagreements.py       # list judge-vs-rule disagreements
```

Re-generating the raw results takes hours on a GPU (`run_pipeline.py`,
`run_multimodel.py`, `run_multidataset.py`). Folder names are historical:
`results_llama3` holds Qwen2-7B and `results_phi3` holds Qwen2-1.5B.
`run_multimodel.py llama3` and `phi3` run those Qwen models.

## Files

| File / folder | Purpose |
|---|---|
| `run_pipeline.py` | Main pipeline: greedy answer, samples, signals |
| `run_multimodel.py`, `run_multidataset.py` | Other models and datasets |
| `run_fingerprinting.py` | Original (thesis) fingerprint analysis, uses labels |
| `analysis.py`, `analysis2.py`, `analysis3.py` | Current analyses (see above) |
| `judge_all.py`, `check_labels.py`, `show_disagreements.py` | Label audit |
| `run_metaclassifier.py` | Learned weighting (see corrections) |
| `run_medqa.py` | Open-ended MedQA run (see corrections) |
| `run_cascade.py`, `run_rag_verification.py`, `run_attention_revisited.py` | Exploratory, thesis-era |
| `app.py`, `start_public.py` | Gradio demo |
| `results*/` | Saved results and plots |
| `analysis_out/`, `analysis_out_judge/` | Current analysis outputs |

## Limitations

200 questions per setting, so intervals are wide; three models in 4-bit
quantization; short factual answers only; correctness labels are imperfect
(containment rule and a single LLM judge); response length was not controlled.

## License

MIT.
