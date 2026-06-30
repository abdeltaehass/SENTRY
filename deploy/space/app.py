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
from model.temporal import (  # noqa: E402
    AGGREGATIONS,
    generate_temporal_with_confidence,
    load_frames,
    temporal_prompt,
)

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
        _STATE.update(model=model, processor=processor, prompt=incident_prompt(cfg),
                      temporal_prompt=temporal_prompt(cfg),
                      num_frames=cfg.get("temporal.num_frames", 4),
                      default_aggregate=cfg.get("temporal.aggregate", "concat"))
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


def _file_paths(files) -> list[str]:
    if not files:
        return []
    files = files if isinstance(files, list) else [files]
    return [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files]


def analyze_clip(files, aggregate):
    paths = _file_paths(files)
    if len(paths) < 2:
        return "Upload at least 2 ordered frames (3–5 works best).", {}, ""
    s = _model()
    aggregate = aggregate or s["default_aggregate"]
    images = load_frames(paths, letterbox_input=True, num_frames=s["num_frames"])
    report, conf = generate_temporal_with_confidence(
        s["model"], s["processor"], images, "cpu",
        max_new_tokens=80, prompt=s["temporal_prompt"], strategy=aggregate, **DECODE,
    )
    rel = assess_reliability(conf)
    return report, {"reliability": rel["reliability_score"]}, _banner(rel)


with gr.Blocks(title="SENTRY — Incident Report Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛰️ SENTRY — Incident Report Generator")
    gr.Markdown(DISCLAIMER)
    with gr.Tabs():
        with gr.Tab("Single frame"):
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(type="pil", label="Surveillance frame", height=360)
                    run = gr.Button("Generate incident report", variant="primary")
                with gr.Column():
                    report_out = gr.Textbox(label="Drafted incident report", lines=6)
                    risk_out = gr.Markdown()
                    conf_out = gr.Label(label="Reliability (0 = unreliable, 1 = reliable)")
            run.click(analyze, inputs=image_in, outputs=[report_out, conf_out, risk_out])

        with gr.Tab("🎞️ Multi-frame (temporal)"):
            gr.Markdown(
                "Upload **3–5 ordered frames** from a short clip — SENTRY encodes each "
                "frame, aggregates across the sequence, and reports the **events over "
                "time** instead of a single scene."
            )
            with gr.Row():
                with gr.Column():
                    frames_in = gr.File(
                        file_count="multiple", type="filepath",
                        label="Ordered frames (3–5)", file_types=["image"],
                    )
                    aggregate_dd = gr.Dropdown(
                        choices=list(AGGREGATIONS), value="concat",
                        label="Cross-frame aggregation",
                    )
                    run_clip = gr.Button("Generate temporal report", variant="primary")
                with gr.Column():
                    clip_report_out = gr.Textbox(label="Temporal incident report", lines=6)
                    clip_risk_out = gr.Markdown()
                    clip_conf_out = gr.Label(label="Reliability (0 = unreliable, 1 = reliable)")
            run_clip.click(analyze_clip, inputs=[frames_in, aggregate_dd],
                           outputs=[clip_report_out, clip_conf_out, clip_risk_out])

if __name__ == "__main__":
    demo.launch()
