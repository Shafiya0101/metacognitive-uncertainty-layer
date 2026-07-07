# Intuition in AI Systems: Reducing Hallucination via Metacognitive Uncertainty Estimation

**PFE Thesis Project | aivancity PGE5 — AI & Data Science | 2025-2026**

---

## The Problem

Large Language Models hallucinate — they generate plausible-sounding but factually incorrect content with full confidence. The model cannot distinguish between what it *knows* and what it is merely *predicting*. This is dangerous in high-stakes domains where users trust AI-generated answers.

## Our Approach

We build a **Metacognitive Uncertainty Layer (MUL)** — a wrapper around a base LLM that estimates *how uncertain* the model is, using four complementary signals:

| Signal | What it measures | How it works |
|--------|-----------------|--------------|
| **Self-Consistency** | Answer agreement | Ask the same question N times; if answers agree, the model likely knows |
| **Semantic Entropy** | Meaning diversity | Cluster answers by *meaning*; more meaning clusters = more uncertainty |
| **Attention Entropy** | Internal focus | Measure how scattered the model's attention is during generation |
| **Logit Calibration** | Token-level confidence | Check how confident the model's internal probability scores are |

## Key Results

| Metric | Value |
|--------|-------|
| Baseline accuracy (no uncertainty layer) | 76.50% |
| Accuracy with fingerprint-aware policy | **93.87%** |
| Coverage (% of questions answered) | 81.5% |
| Accuracy improvement | **+17.37%** |

### Hallucination Detection Performance

![AUROC Comparison](results/auroc_comparison.png)

*Our combined approach (AUROC = 0.7953) outperforms SelfCheckGPT-NLI (0.7924) and all individual methods.*

### Calibration Analysis

![Calibration Diagram](results/calibration_diagram.png)

*The model is overconfident: ECE = 0.1276. Internal confidence scores cannot be trusted, motivating external uncertainty estimation.*

### Accuracy-Coverage Tradeoff

![Accuracy Coverage](results/accuracy_coverage.png)

*By abstaining on uncertain questions, accuracy improves dramatically while still answering 81.5% of questions.*

### Uncertainty Score Distributions

![Uncertainty Distributions](results/uncertainty_distributions.png)

*Each method's score distribution for correct (green) vs hallucinated (red) answers. Better separation = better detection.*

---

## Novel Contribution: Uncertainty Fingerprinting

Existing methods treat hallucination as binary. We introduce **uncertainty fingerprinting** — classifying hallucinations into distinct *types* based on the signal pattern:

![Fingerprint Distribution](results/fingerprint_distribution.png)

| Fingerprint Type | Signal Pattern | Danger Level | Response |
|-----------------|---------------|-------------|----------|
| **Confident Fabrication** | All signals say "confident" but answer is wrong | Highest | Flag for verification |
| **Knowledge Gap** | All signals show high uncertainty | Lowest | Abstain |
| **Shallow Mimicry** | Text agrees but meaning diverges | Medium | Verify details |

### Signal Fingerprint Heatmap

![Fingerprint Heatmap](results/fingerprint_heatmap.png)

*Each hallucination type has a distinct signal pattern — this is what makes type-specific response strategies possible.*

### Novel Contribution Comparison

![Novel Contribution](results/novel_contribution.png)

*Our fingerprint-enhanced method outperforms both equal-weight combination and the learned meta-classifier.*

---

## Live Demo

The interactive Gradio app lets you type any question and see the full uncertainty analysis in real-time:

- AI answer with confidence assessment
- Four uncertainty signal bars (color-coded green/orange/red)
- Hallucination fingerprint classification
- Actionable recommendation (trust / verify / abstain)

```bash
python app.py  # Opens at http://localhost:7860
```

---

## Architecture

![Architecture](results/architecture.png)

## Repository Structure

| File | Description |
|------|-------------|
| `run_pipeline.py` | Main experimental pipeline — all 5 steps (~4 hours) |
| `run_fingerprinting.py` | Uncertainty fingerprinting analysis (~1 minute) |
| `app.py` | Gradio interactive demo application |
| `results/` | All evaluation plots |

## Tech Stack

- **Base Model:** Mistral-7B-Instruct (4-bit quantized via bitsandbytes)
- **NLI Model:** DeBERTa-Large-MNLI
- **Dataset:** TriviaQA (200 questions, 10 samples each)
- **Framework:** PyTorch, HuggingFace Transformers, scikit-learn
- **Demo:** Gradio

## Setup & Run

```bash
git clone https://github.com/Shafiya0101/metacognitive-uncertainty-layer.git
cd metacognitive-uncertainty-layer
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate bitsandbytes datasets sentencepiece protobuf scipy scikit-learn matplotlib tqdm gradio

python run_pipeline.py        # Full pipeline (~4 hours, needs GPU)
python run_fingerprinting.py  # Fingerprinting (~1 minute)
python app.py                 # Launch demo at localhost:7860
```

**Requirements:** Python 3.10+, NVIDIA GPU 8GB+ VRAM, ~15GB disk for model weights.

## Reference Papers

1. Huang et al. (2023) — *A Survey on Hallucination in Large Language Models*
2. Wang et al. (2023) — *Self-Consistency Improves Chain of Thought Reasoning* (ICLR)
3. Farquhar et al. (2024) — *Detecting Hallucinations Using Semantic Entropy* (Nature)
4. Guo et al. (2017) — *On Calibration of Modern Neural Networks* (ICML)
5. Manakul et al. (2023) — *SelfCheckGPT: Zero-Resource Hallucination Detection* (EMNLP)

## Author

**Shafiya Kausar** — MSc AI & Data Science, aivancity Paris

*Presented at PyCon ES 2026, Barcelona*
