"""Hugging Face Space entrypoint — SENTRY incident report generator.

Self-contained: the `src/` tree is vendored alongside this file. Loads the model
(a trained adapter if `SENTRY_ADAPTER` is set, else the base VLM), int8-quantized
for free CPU, and serves an upload-frame -> incident report + reliability demo.
Grad-CAM grounding needs gradients (disabled under quantization) — run it locally
on GPU/MPS for the grounding overlay.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "src")

import gradio as gr  # noqa: E402

from config import load_config  # noqa: E402
from eval.reliability import assess_reliability  # noqa: E402
from model.inference import generate_with_confidence  # noqa: E402
from model.model import load_for_demo  # noqa: E402
from model.prompts import incident_prompt  # noqa: E402

CONFIG = "configs/default.yaml"
ADAPTER = os.environ.get("SENTRY_ADAPTER", "")   # empty -> base model
DECODE = dict(num_beams=1, repetition_penalty=1.5, no_repeat_ngram_size=3)  # greedy = faster on CPU

DISCLAIMER = (
    "Upload a surveillance frame to get a drafted incident report and a "
    "**reliability score** — high-risk outputs are flagged for human review.\n\n"
    "⚠️ *Research demo — not for real security decisions. Running the "
    "prompt-conditioned base model on free CPU, so the first request is slow.*"
)
_STATE: dict = {}


def _model():
    if "model" not in _STATE:
        import torch
        torch.set_num_threads(max(1, os.cpu_count() or 2))
        cfg = load_config(CONFIG)
        model, processor = load_for_demo(cfg, ADAPTER or None, device="cpu", quantize=True)
        _STATE.update(model=model, processor=processor, prompt=incident_prompt(cfg))
    return _STATE


def _banner(rel: dict) -> str:
    s = rel["reliability_score"]
    if rel["flagged"]:
        return f"### 🛑 FLAGGED — high hallucination risk (reliability {s:.2f}) — verify before acting"
    if rel["risk_level"] == "elevated":
        return f"### ⚠️ Elevated risk (reliability {s:.2f}) — review recommended"
    return f"### ✅ Low risk (reliability {s:.2f})"


def analyze(image):
    if image is None:
        return "Please upload a frame.", {}, ""
    s = _model()
    report, conf = generate_with_confidence(
        s["model"], s["processor"], image.convert("RGB"), "cpu",
        max_new_tokens=80, prompt=s["prompt"], **DECODE,
    )
    rel = assess_reliability(conf)
    return report, {"reliability": rel["reliability_score"]}, _banner(rel)


with gr.Blocks(title="SENTRY — Incident Report Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛰️ SENTRY — Incident Report Generator")
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Surveillance frame", height=360)
            run = gr.Button("Generate incident report", variant="primary")
        with gr.Column():
            report_out = gr.Textbox(label="Drafted incident report", lines=6)
            risk_out = gr.Markdown()
            conf_out = gr.Label(label="Reliability score (0 = unreliable, 1 = reliable)")
    run.click(analyze, inputs=image_in, outputs=[report_out, conf_out, risk_out])

if __name__ == "__main__":
    demo.launch()
