import gradio as gr
import torch
import numpy as np
import re, string, traceback
from collections import Counter

print("Loading model...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer as ST

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=qc, device_map="auto", dtype=torch.float16)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
print("Loading NLI model...", flush=True)
NLI_MODEL = "microsoft/deberta-large-mnli"
nli_tokenizer = ST.from_pretrained(NLI_MODEL)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
try:
    nli_model = nli_model.to("cuda")
    nli_device = "cuda"
except:
    nli_device = "cpu"
print("All models loaded!", flush=True)
N_SAMPLES = 5

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

def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def check_entailment(a, b):
    def _e(p, h):
        inp = nli_tokenizer(p, h, return_tensors="pt", truncation=True, max_length=256).to(nli_device)
        with torch.no_grad():
            return nli_model(**inp).logits.argmax(-1).item() == 2
    return _e(a, b) and _e(b, a)

def selfcheck_nli(main_a, samp_a):
    prem = "The answer to the question is: " + samp_a
    hyp = "The answer to the question is: " + main_a
    inp = nli_tokenizer(prem, hyp, return_tensors="pt", truncation=True, max_length=256).to(nli_device)
    with torch.no_grad():
        logits = nli_model(**inp).logits
    ze, zc = logits[0, 2].item(), logits[0, 0].item()
    return np.exp(zc) / (np.exp(ze) + np.exp(zc))

def make_bar(value, max_val, label, score_text):
    pct = min(max(value / max_val, 0), 1.0) * 100
    if pct < 30:
        color = "#22c55e"
    elif pct < 60:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    return f'''<div style="margin-bottom:16px">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px">
        <span style="color:#e2e8f0;font-weight:600;font-size:14px">{label}</span>
        <span style="color:{color};font-weight:700;font-size:14px">{score_text}</span>
      </div>
      <div style="background:#1e293b;border-radius:10px;height:14px;overflow:hidden">
        <div style="background:{color};height:100%;border-radius:10px;width:{pct}%"></div>
      </div>
    </div>'''

def analyze(question):
    if not question.strip():
        return ('<p style="color:#94a3b8;text-align:center;padding:60px 20px;font-size:16px">'
                'Type a question above and click Analyze</p>'), "", ""
    try:
        prompt = format_prompt(question)
        greedy, lps = generate(prompt, temp=0.0)
        avg_nlp = -np.mean(lps) if lps else 0.0

        samples = []
        for i in range(N_SAMPLES):
            r, _ = generate(prompt, temp=0.7)
            samples.append(r)

        normed = [normalize_answer(s) for s in samples]
        cts = Counter(normed)
        maj, majc = cts.most_common(1)[0]
        sc_unc = 1.0 - majc / N_SAMPLES

        clusters = []
        for resp in samples:
            if not resp.strip():
                continue
            placed = False
            for cl in clusters:
                if check_entailment(cl[0], resp):
                    cl.append(resp)
                    placed = True
                    break
            if not placed:
                clusters.append([resp])
        probs_arr = np.array([len(c) / N_SAMPLES for c in clusters])
        se = float(-np.sum(probs_arr * np.log(probs_arr + 1e-12)))

        nli_sc = [selfcheck_nli(greedy, s) if s.strip() and greedy.strip() else 0.5 for s in samples]
        selfcheck = float(np.mean(nli_sc))

        sc_h = sc_unc > 0.3
        se_h = se > 0.5
        if not sc_h and not se_h:
            fp = "HIGH CONFIDENCE"
            fp_color = "#22c55e"
            fp_bg = "rgba(34,197,94,0.1)"
            rec_label = "LIKELY RELIABLE"
            rec = "All uncertainty signals are low. The model consistently produced the same answer with the same meaning. This answer is likely correct, but always verify critical facts."
        elif sc_h and se_h:
            fp = "KNOWLEDGE GAP"
            fp_color = "#ef4444"
            fp_bg = "rgba(239,68,68,0.1)"
            rec_label = "DO NOT TRUST"
            rec = "The model clearly does not know the answer. It produced different answers with different meanings each time. This is a hallucination."
        elif not sc_h and se_h:
            fp = "SHALLOW MIMICRY"
            fp_color = "#f59e0b"
            fp_bg = "rgba(245,158,11,0.1)"
            rec_label = "VERIFY CAREFULLY"
            rec = "The answers look similar on the surface but differ in meaning. The model may be mimicking patterns without genuine understanding."
        else:
            fp = "MIXED SIGNALS"
            fp_color = "#8b5cf6"
            fp_bg = "rgba(139,92,246,0.1)"
            rec_label = "USE WITH CAUTION"
            rec = "Some uncertainty signals are elevated while others are low. The answer may be partially correct. Verify important details."

        n_unique = len(set(normed))
        n_clusters = len(clusters)

        main_html = f'''<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">

          <div style="background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #334155;border-radius:16px;padding:28px;margin-bottom:18px">
            <div style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;font-weight:700">AI Answer</div>
            <div style="color:#f1f5f9;font-size:22px;font-weight:700;line-height:1.5">{greedy}</div>
          </div>

          <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-bottom:18px">
            <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;padding:18px;text-align:center">
              <div style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:1.5px;font-weight:700">Samples</div>
              <div style="color:#f1f5f9;font-size:28px;font-weight:800;margin-top:6px">{N_SAMPLES}</div>
            </div>
            <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;padding:18px;text-align:center">
              <div style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:1.5px;font-weight:700">Clusters</div>
              <div style="color:#f1f5f9;font-size:28px;font-weight:800;margin-top:6px">{n_clusters}</div>
            </div>
            <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;padding:18px;text-align:center">
              <div style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:1.5px;font-weight:700">Unique</div>
              <div style="color:#f1f5f9;font-size:28px;font-weight:800;margin-top:6px">{n_unique}</div>
            </div>
            <div style="background:{fp_bg};border:2px solid {fp_color};border-radius:12px;padding:18px;text-align:center">
              <div style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:1.5px;font-weight:700">Verdict</div>
              <div style="color:{fp_color};font-size:13px;font-weight:800;margin-top:8px">{fp}</div>
            </div>
          </div>

          <div style="background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #334155;border-radius:16px;padding:28px;margin-bottom:18px">
            <div style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:2px;margin-bottom:20px;font-weight:700">Uncertainty Signals</div>
            {make_bar(sc_unc, 1.0, "Self-Consistency", f"{sc_unc:.3f}")}
            {make_bar(se, 2.3, "Semantic Entropy", f"{se:.3f}")}
            {make_bar(avg_nlp, 0.5, "Avg(-logP)", f"{avg_nlp:.3f}")}
            {make_bar(selfcheck, 1.0, "SelfCheckGPT-NLI", f"{selfcheck:.3f}")}
          </div>

          <div style="background:{fp_bg};border:2px solid {fp_color};border-radius:16px;padding:32px;margin-bottom:18px;text-align:center">
            <div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:2px;margin-bottom:16px;font-weight:700">Hallucination Fingerprint</div>
            <div style="display:inline-block;background:{fp_color};color:white;font-size:20px;font-weight:800;padding:14px 44px;border-radius:50px;letter-spacing:2px;box-shadow:0 4px 20px {fp_color}55">{fp}</div>
          </div>

          <div style="background:linear-gradient(135deg,#0f172a,#1e293b);border-left:5px solid {fp_color};border-radius:0 16px 16px 0;padding:24px 28px">
            <div style="color:{fp_color};font-size:16px;font-weight:800;margin-bottom:10px;letter-spacing:0.5px">{rec_label}</div>
            <div style="color:#cbd5e1;font-size:15px;line-height:1.7">{rec}</div>
          </div>

        </div>'''

        samples_html = f'''<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
          <div style="background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #334155;border-radius:16px;padding:24px">
            <div style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:2px;margin-bottom:16px;font-weight:700">Generated Samples (Temperature = 0.7)</div>'''

        for i, s in enumerate(samples):
            n = normalize_answer(s)
            matches = n == normalize_answer(greedy)
            dot = "#22c55e" if matches else "#ef4444"
            tag = "matches" if matches else "differs"
            samples_html += f'''<div style="display:flex;align-items:flex-start;gap:12px;padding:12px 0;{"border-bottom:1px solid #1e293b" if i < len(samples)-1 else ""}">
              <div style="background:{dot};width:10px;height:10px;border-radius:50%;margin-top:5px;flex-shrink:0"></div>
              <div style="flex:1">
                <span style="color:#64748b;font-size:11px;font-weight:600">Sample {i+1} &middot; {tag}</span>
                <div style="color:#e2e8f0;font-size:14px;margin-top:3px">{s}</div>
              </div>
            </div>'''

        samples_html += '''<div style="color:#475569;font-size:11px;margin-top:14px">
            <span style="color:#22c55e">&#9679;</span> matches greedy answer &nbsp;&nbsp;
            <span style="color:#ef4444">&#9679;</span> different answer
          </div></div></div>'''

        return main_html, samples_html, ""

    except Exception as e:
        traceback.print_exc()
        return f"<p style='color:#ef4444;padding:20px'>Error: {e}</p>", "", ""

CSS = """
.gradio-container {background:#020617 !important; max-width:1400px !important}
footer {display:none !important}
"""

with gr.Blocks(title="Metacognitive Uncertainty Layer", css=CSS) as demo:
    gr.HTML('''<div style="text-align:center;padding:28px 0 10px 0">
        <h1 style="color:#f1f5f9;font-size:36px;font-weight:800;margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">Metacognitive Uncertainty Layer</h1>
        <p style="color:#64748b;font-size:14px;margin-top:8px;font-family:-apple-system,sans-serif">Hallucination Detection via Uncertainty Fingerprinting &bull; PFE Thesis &bull; Shafiya Kausar</p>
    </div>''')

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320):
            question = gr.Textbox(label="Ask a question", placeholder="Type any factual question...", lines=3)
            btn = gr.Button("Analyze", variant="primary", size="lg")
            gr.Examples(
                examples=[
                    ["What is the capital of Australia?"],
                    ["Who painted the Mona Lisa?"],
                    ["Who was the first person to walk on Mars?"],
                    ["Who won the Nobel Prize in Computer Science in 2020?"],
                    ["What year did Leonardo da Vinci paint the Starry Night?"],
                    ["How many moons does Mercury have?"],
                    ["What is the chemical symbol for gold?"],
                ],
                inputs=question,
                label="Example Questions",
            )

        with gr.Column(scale=3, min_width=600):
            output = gr.HTML(value='<p style="color:#94a3b8;text-align:center;padding:60px 20px;font-size:16px">Type a question and click Analyze</p>')

    with gr.Accordion("View Raw Samples", open=False):
        raw = gr.HTML()

    hidden = gr.HTML(visible=False)

    gr.HTML('<div style="text-align:center;padding:20px;color:#334155;font-size:11px">Model: Mistral-7B-Instruct &bull; Self-Consistency + Semantic Entropy + Logit Calibration + SelfCheckGPT-NLI</div>')

    btn.click(analyze, inputs=question, outputs=[output, raw, hidden])

demo.launch(share=True, server_name="127.0.0.1", server_port=7860)